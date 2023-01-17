!pip install -q wtte
import numpy as np

import pandas as pd

from tqdm.auto import tqdm



import keras

import wtte.wtte as wtte



from tensorflow import set_random_seed

set_random_seed(5944)



%matplotlib inline

import matplotlib.pyplot as plt



K = keras.backend
def _keras_unstack_hack(ab):

    """Implements tf.unstack(y_true_keras, num=2, axis=-1).

       Keras-hack adopted to be compatible with theano backend.

    """

    ndim = len(K.int_shape(ab))

    if ndim == 0:

        print('can not unstack with ndim=0')

    else:

        a = ab[..., 0]

        b = ab[..., 1]

    return a, b



def weibull_loglik_discrete(y_true, ab_pred, name=None):

    y_ = y_true[:, 0]

    u_ = y_true[:, 1]

    a_ = ab_pred[:, 0]

    b_ = ab_pred[:, 1]



    hazard0 = K.pow((y_ + 1e-35) / a_, b_)

    hazard1 = K.pow((y_ + 1) / a_, b_)



    return -1 * K.mean(u_ * K.log(k.exp(hazard1 - hazard0) - 1.0) - hazard1)



"""

    Not used for this model, but included in case somebody needs it

    For math, see https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf (Page 35)

"""

def weibull_loglik_continuous(y_true, ab_pred, name=None):

    y_ = y_true[:, 0]

    u_ = y_true[:, 1]

    a_ = ab_pred[:, 0]

    b_ = ab_pred[:, 1]



    ya = (y_ + 1e-35) / a_

    return -1 * K.mean(u_ * (K.log(b_) + b_ * K.log(ya)) - K.pow(ya, b_))





"""

    Custom Keras activation function, outputs alpha neuron using exponentiation and beta using softplus

"""

def activate(ab):

    a = K.exp(ab[:, 0])

    b = K.softplus(ab[:, 1])



    a = K.reshape(a, (K.shape(a)[0], 1))

    b = K.reshape(b, (K.shape(b)[0], 1))



    return K.concatenate((a, b), axis=1)



def output_lambda(x, init_alpha=1.0, max_beta_value=5.0, max_alpha_value=None):

    """Elementwise (Lambda) computation of alpha and regularized beta.



        Alpha: 

        (activation) 

        Exponential units seems to give faster training than 

        the original papers softplus units. Makes sense due to logarithmic

        effect of change in alpha. 

        (initialization) 

        To get faster training and fewer exploding gradients,

        initialize alpha to be around its scale when beta is around 1.0,

        approx the expected value/mean of training tte. 

        Because we're lazy we want the correct scale of output built

        into the model so initialize implicitly; 

        multiply assumed exp(0)=1 by scale factor `init_alpha`.



        Beta: 

        (activation) 

        We want slow changes when beta-> 0 so Softplus made sense in the original 

        paper but we get similar effect with sigmoid. It also has nice features.

        (regularization) Use max_beta_value to implicitly regularize the model

        (initialization) Fixed to begin moving slowly around 1.0



        Assumes tensorflow backend.



        Args:

            x: tensor with last dimension having length 2

                with x[...,0] = alpha, x[...,1] = beta



        Usage:

            model.add(Dense(2))

            model.add(Lambda(output_lambda, arguments={"init_alpha":100., "max_beta_value":2.0}))

        Returns:

            A positive `Tensor` of same shape as input

    """

    a, b = _keras_unstack_hack(x)



    # Implicitly initialize alpha:

    if max_alpha_value is None:

        a = init_alpha * K.exp(a)

    else:

        a = init_alpha * K.clip(x=a, min_value=K.epsilon(),

                                max_value=max_alpha_value)



    m = max_beta_value

    if m > 1.05:  # some value >>1.0

        # shift to start around 1.0

        # assuming input is around 0.0

        _shift = np.log(m - 1.0)



        b = K.sigmoid(b - _shift)

    else:

        b = K.sigmoid(b)



    # Clipped sigmoid : has zero gradient at 0,1

    # Reduces the small tendency of instability after long training

    # by zeroing gradient.

    b = m * K.clip(x=b, min_value=K.epsilon(), max_value=1. - K.epsilon())



    x = K.stack([a, b], axis=-1)



    return x
def extract_features(z):

     return np.c_[z.mean(axis=1), 

                  z.min(axis=1),

                  z.max(axis=1),

                  z.std(axis=1)]



def create_X(x, last_index=None, n_steps=150, step_length=1000):

    if last_index == None:

        last_index=len(x)

       

    assert last_index - n_steps * step_length >= 0



    # Reshaping and approximate standardization with mean 5 and std 3.

    temp = (x[(last_index - n_steps * step_length):last_index].reshape(n_steps, -1) - 5 ) / 3

    

    # Extracts features of sequences of full length 1000, of the last 100 values and finally also 

    # of the last 10 observations. 

    return np.c_[extract_features(temp),

                 extract_features(temp[:, -step_length // 10:]),

                 extract_features(temp[:, -step_length // 100:])]





def weibull_loss_discrete(y_true, y_pred, name=None):

    """calculates a keras loss op designed for the sequential api.

    

        Discrete log-likelihood for Weibull hazard function on censored survival data.

        For math, see 

        https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf (Page 35)

        

        Args:

            y_true: tensor with last dimension having length 2

                with y_true[:,...,0] = time to event, 

                     y_true[:,...,1] = indicator of not censored

                

            y_pred: tensor with last dimension having length 2 

                with y_pred[:,...,0] = alpha, 

                     y_pred[:,...,1] = beta



        Returns:

            A positive `Tensor` of same shape as input

            

    """    

    y,u = _keras_unstack_hack(y_true)

    a,b = _keras_unstack_hack(y_pred)



    hazard0 = K.pow((y + 1e-35) / a, b)

    hazard1 = K.pow((y + 1.0) / a, b)

    

    loglikelihoods = u * K.log(K.exp(hazard1 - hazard0) - 1.0) - hazard1

    loss = -1 * K.mean(loglikelihoods)

    return loss
float_data = pd.read_csv('../input/train.csv',

                         dtype={'acoustic_data': np.float32, 'time_to_failure': np.float32}).values
# Query "create_X" to figure out the number of features

n_features = create_X(float_data[0:150000]).shape[1]

print("Our RNN is based on %i features"% n_features)
def generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150, step_length=1000, test=False):

    if max_index is None:

        max_index = len(data) - 1

     

    while True:

        # Pick indices of ending positions

        rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size)

         

        # Initialize feature matrices and targets

        samples = np.zeros((batch_size, n_steps, n_features))

        targets = np.zeros((batch_size, 2))

        

        for j, row in enumerate(rows):

            samples[j] = create_X(data[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)

            targets[j, 0] = data[row - 1, 1]

            if test:

                targets[j, 1] = 1

            else:

                targets[j, 1] = 1 if targets[j, 0] > 0.03 else 0

        yield samples, targets
# Position of second (of 16) earthquake. Used to have a clean split

# between train and validation

second_earthquake = 50085877

float_data[second_earthquake, 1]



batch_size = 32

mask_value = -99

init_alpha = -1.0 / np.log(1.0 - 1.0/ (5 + 1.0) )

init_alpha = init_alpha / n_features



# Initialize generators

train_gen = generator(float_data, batch_size=batch_size) # Use this for better score

# train_gen = generator(float_data, batch_size=batch_size, min_index=second_earthquake + 1)

valid_gen = generator(float_data, batch_size=batch_size, max_index=second_earthquake, test=True)
aux, aux2 = next(train_gen)

print(aux.shape, aux2.shape)
def weibull_mode(alpha, beta):

    assert np.all(beta > 1)

    return alpha * ((beta-1)/beta)**(1/beta)



def weibull_quantiles(alpha, beta, p):

    return alpha*np.power(-np.log(1.0-p),1.0/beta)



def weibull_mode_K(a, b):

    #assert np.all(beta > 1)

    return a * K.pow((b - 1) / b, (1 / b))
def mode_mae(y_true, y_pred, name=None):

    y, u = _keras_unstack_hack(y_true)

    a, b = _keras_unstack_hack(y_pred)

    pred = weibull_mode_K(a, b)

    

    return K.mean(K.abs(pred - y), axis=-1)
# callbacks

history = keras.callbacks.History()

ww = wtte.WeightWatcher()

nanterminator = keras.callbacks.TerminateOnNaN()

cb = [history, ww, nanterminator]



# Start building our model

model = keras.Sequential()



# Mask parts of the lookback period that are all zeros (i.e., unobserved) so they don't skew the model

#model.add(tfkl.Masking(mask_value=mask_value, input_shape=(None, n_features)))

model.add(keras.layers.Conv1D(64, kernel_size=32, input_shape=(None, n_features)))

model.add(keras.layers.Conv1D(64, kernel_size=16))

model.add(keras.layers.MaxPool1D(2))



model.add(keras.layers.Conv1D(128, kernel_size=16))

model.add(keras.layers.Conv1D(128, kernel_size=12))

model.add(keras.layers.MaxPool1D(2))



# recurrent layer(s)

model.add(keras.layers.GRU(50, activation='tanh', recurrent_dropout=0.25, dropout=0.3, input_shape=(None, n_features)))

#model.add(keras.layers.CuDNNLSTM(100))

#model.add(keras.layers.CuDNNGRU(30, input_shape=(None, n_features)))



# We need 2 neurons to output Alpha and Beta parameters for our Weibull distribution



# model.add(keras.layers.TimeDistributed(tfkl.Dense(2)))

model.add(keras.layers.Dense(2))



# Apply the custom activation function mentioned above

model.add(keras.layers.Activation(activate))



model.add(keras.layers.Lambda(

    wtte.output_lambda, 

    arguments={'init_alpha': 3, 

               'max_beta_value': 500.0, 

               'scalefactor': 0.5,}))



# Use the discrete log-likelihood for Weibull survival data as our loss function

loss = wtte.loss(kind='discrete', reduce_loss=False).loss_function



model.compile(loss=loss,

              optimizer=keras.optimizers.SGD(lr=.01, clipnorm=2),

              metrics=[mode_mae])
steps_train = 4196 // batch_size

steps_test = steps_train // 2



history = model.fit_generator(train_gen,

                              steps_per_epoch=1000,

                              epochs=50,

                              verbose=1,

                              callbacks=cb,

                              validation_data=valid_gen,

                              validation_steps=200)
plt.plot(history.history['loss'], label='training')

plt.plot(history.history['val_loss'],label='validation')

plt.legend();
ww.plot()
np.array(y_train_true).shape
from tqdm.auto import tqdm



y_train_true = []

y_train_pred = []

y_train_pred_25 = []

y_train_pred_75 = []



valid_gen = generator(float_data, batch_size=batch_size, max_index=second_earthquake)





for _ in tqdm(range(500)):

    feats, tgt = next(train_gen)

    y_train_true.append(tgt[-1, 0])

    preds = model.predict(feats)

    alpha_pred, beta_pred = preds[-1, 0], preds[-1, 1]

    y_train_pred.append(weibull_mode(alpha_pred, beta_pred))

    y_train_pred_25.append(weibull_quantiles(alpha_pred, beta_pred, 0.25))

    y_train_pred_75.append(weibull_quantiles(alpha_pred, beta_pred, 0.75))



plt.figure(figsize=(9, 9))

plt.scatter(np.array(y_train_true), np.array(y_train_pred))

plt.xlabel('ytrue')

plt.ylabel('ypred');
from sklearn.metrics import mean_absolute_error



mean_absolute_error(y_train_true, y_train_pred)
from tqdm.auto import tqdm



# Load submission file

submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})



# Load each test data, create the feature matrix, get numeric prediction

for i, seg_id in enumerate(tqdm(submission.index)):

  #  print(i)

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    x = seg['acoustic_data'].values

    preds = model.predict(np.expand_dims(create_X(x), 0))

    alpha_pred, beta_pred = preds[-1, 0], preds[-1, 1]

    submission.time_to_failure[i] = weibull_mode(alpha_pred, beta_pred)
submission.head()
submission.to_csv('submission.csv')