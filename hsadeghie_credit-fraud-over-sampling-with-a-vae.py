from os.path import exists

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve

import numpy as np

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score

import pickle

import matplotlib.pyplot as plt
def get_scaled_data_splitted():

    df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

    std_scaler = StandardScaler()

    columns = df.columns.drop(['Class'])

    for col in columns:

        df['scaled_' + str(col)] = std_scaler.fit_transform(df[col].values.reshape(-1, 1))

        df.drop(col, axis=1, inplace=True)

    X = df.drop('Class', axis=1)

    y = df['Class']

    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

    return pd.concat((xtrain, ytrain), axis=1), pd.concat((xtest, ytest), axis=1)
def fraud_or_not(dftrain, dftest):

    fraud = dftrain.loc[dftrain['Class'] == 1].sample(frac=1)

    fraudtest = dftest.loc[dftest['Class'] == 1].sample(frac=1)

    return fraud.drop('Class', axis=1), fraudtest.drop('Class', axis=1)
import tensorflow as tf

from tensorflow.keras.layers import InputLayer, Dense, Flatten, Reshape

from tensorflow.keras import Sequential

import time

import numpy as np



# This code is taken from Tensorflow tutorial on VAEs. It's turned from a CNN to a simple neuralnet which is more appropriate for our case here.



# almost the standard for activation these days

relu = tf.nn.relu





class VAE(tf.keras.Model):

    def __init__(self, ndim, latent_dim):

        super(VAE, self).__init__()

        self.latent_dim = latent_dim  # number of latent variables of the VAE

        

        # a simple neural net with one hidden layer. At the end we produce twice the number of latent variables because we are modeling the mean and variance of the Gaussian random variates.

        self.inference_net = Sequential(

            [

                InputLayer(input_shape=(ndim,)),

                Dense(100, activation=relu),

                Dense(2 * latent_dim)

            ]

        )

        

        # A simple decoder network at produces the mean of the variables. An improvement would be to model the variance as well. The training becomes tricky since the variance can collapse.

        self.generative_net = Sequential(

            [

                InputLayer(input_shape=(latent_dim,)),

                Dense(100, activation=relu),

                Dense(ndim)

            ])



    # We need this function to generate samples. Start with random normal noise. pass it through the decoder, and you get your samples

    @tf.function

    def sample(self, num_samples=100, eps=None):

        if eps is None:

            eps = tf.random.normal(shape=(num_samples, self.latent_dim))

        return self.decode(eps)



    # a function to call the encoder network

    def encode(self, x):

        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)

        return mean, logvar



    # a function to reparameterie the encoder for the ease of backpropagation

    def reparameterize(self, mean, logvar):

        eps = tf.random.normal(shape=mean.shape)

        return eps * tf.exp(logvar * 0.5) + mean



    # a function to call the decoder

    def decode(self, z):

        return self.generative_net(z)

# a function for computing the KL term of Gaussian distribution

def log_normal_pdf(sample, mean, logvar, raxis=1):

    log2pi = tf.math.log(2. * np.pi)

    return tf.reduce_sum(-0.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),

                         axis=raxis)



# a function to compute the loss of the VAE

@tf.function

def compute_loss(model, x):

    mean, logvar = model.encode(x)

    logvar = tf.clip_by_value(logvar, -88., 88.)

    z = model.reparameterize(mean, logvar)

    xmean = model.decode(z)

    logpx_z = -tf.reduce_sum((x - xmean) ** 2, axis=1)  # ad-hoc l2 loss that is pretty close to log-prob of a gaussian distribution withtout taking into account the variance

    logpz = log_normal_pdf(z, 0.0, 0.0)

    logqz_x = log_normal_pdf(z, mean, logvar)

    return -tf.reduce_mean(logpx_z + logpz - logqz_x)



# A function that given the model computes the loss, the gradients and apply the parameter update

@tf.function

def compute_apply_gradients(model, x, optimizer):

    with tf.GradientTape() as tape:

        loss = compute_loss(model, x)

    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
def train(xtrain, xtest, model=None, load=False, filepath=None):

    optimizer = tf.keras.optimizers.Adam(1e-3)

    epochs = 2000

    latent_dim = 20

    num_train, ndim = xtrain.shape

    num_test, _ = xtest.shape

    if model is None:

        model = VAE(ndim, latent_dim)

    if load and filepath is not None:

        model.load_weights(filepath=filepath)

        return model

    else:

        batch_size = 32

        train_dataset = tf.data.Dataset.from_tensor_slices(xtrain.values.astype(np.float32)).shuffle(num_train).batch(

            batch_size)



        test_dataset = tf.data.Dataset.from_tensor_slices(xtest.values.astype(np.float32)).shuffle(num_test).batch(num_test)



        for epoch in range(1, epochs + 1):

            start_time = time.time()

            for train_x in train_dataset:

                compute_apply_gradients(model, train_x, optimizer)

            end_time = time.time()



            if epoch % 100 == 0:

                loss = tf.keras.metrics.Mean()

                for test_x in test_dataset:

                    loss(compute_loss(model, test_x))

                elbo = -loss.result()

                print('Epoch: {}, Test set psudo-ELBO: {}, '

                      'time elapse for current epoch {}'.format(epoch, elbo, end_time - start_time))

                model.save_weights('saved_models/model_%d_at_%d' % (latent_dim, epoch))

    return model
def augment_data(data, model):

    num_samples = data['Class'].value_counts()[0] - data['Class'].value_counts()[1]

    samples = model.sample(num_samples=num_samples).numpy()

    dfnew = pd.DataFrame(samples, columns=data.columns.drop('Class'))

    dfnew['Class'] = np.ones(len(samples), dtype=np.int)

    dfnew = pd.concat((data, dfnew), ignore_index=True).sample(frac=1)

    return dfnew
# get the data and split it

dtrain, dtest = get_scaled_data_splitted()

# get the fraud cases from train and test

Xf, Xft = fraud_or_not(dtrain, dtest)

# get the traied VAE model

model = train(Xf, Xft, load=False, filepath='saved_models/model_20_at_1900')

# augment the data using the VAE model

augmented = augment_data(dtrain, model)

X = augmented.drop('Class', axis=1)

y = augmented['Class']

Xt = dtest.drop('Class', axis=1)

yt = dtest['Class']
classifiers = {

    "XGBClassifier": XGBClassifier(),

    "BaggingClassifier": BaggingClassifier(),

    "RandomForestClassifier": RandomForestClassifier(),

}



N = 10000

for key, classifier in classifiers.items():

    print('_' * 50)

    name = key

    print(name)

    if exists(name):

        print('loading...')

        classifier = pickle.load(open(name, 'rb'))

        training_score = cross_val_score(classifier, X[:N], y[:N], cv=5)

        print("Classifiers: ", name, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")

    else:

        classifier.fit(X, y)

        print('validating ...')

        training_score = cross_val_score(classifier, X[:N], y[:N], cv=5)

        print("Classifiers: ", name, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")

        pickle.dump(classifier, open(name, 'wb'))

    print(classifier.score(X, y))

    print(classifier.score(Xt, yt))

    y_pred = classifier.predict(Xt)

    cm = confusion_matrix(yt, y_pred)

    print(cm)

    print(classification_report(yt, y_pred))

    print(average_precision_score(yt, y_pred))
