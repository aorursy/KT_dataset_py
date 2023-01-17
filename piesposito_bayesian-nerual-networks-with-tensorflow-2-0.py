#%pip install -U pip

%pip install tensorflow==2.0

%pip install tfp-nightly
import tensorflow as tf

import tensorflow_probability as tfp

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import time

from sklearn.model_selection import train_test_split



%matplotlib inline

#random seed as the birthday of my granp which is in the hospital fighting with cancer

#be strong Valdomiro!

np.random.seed(10171927)

tf.random.set_seed(10171927)



#to see how long the notebook lasts to run

start = time.time()
print('TensorFlow version (expected = 2.0.0):', tf.__version__)

print('TensorFlow Probability version (expected = 0.9.0-dev20190912):', tfp.__version__)
#We first load our data, and seek the distribution for its labels

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")



#Create our labels array

Y_train = train["label"]



# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 



#Visualize the distribution 

g = sns.countplot(Y_train)
X_train = X_train / 255.0

X_train = X_train.values.reshape(-1,28,28,1)

Y_train = tf.keras.utils.to_categorical(Y_train, num_classes = 10)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.3, random_state=42)
#We are using 1026, as it is my birthday

idx = 1026

plt.imshow(X_train[idx, :, :, 0], cmap='gist_gray')

print("True label of the test sample {}: {}".format(idx, np.argmax(Y_train[idx], axis=-1)))
def build_cnn(input_shape):

    

    ##model building

    model = tf.keras.models.Sequential()

    #convolutional layer with rectified linear unit activation

    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),

                     activation='relu',

                     input_shape=input_shape))

    #32 convolution filters used each of size 3x3

    #again

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

    #64 convolution filters used each of size 3x3

    #choose the best features via pooling

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    #randomly turn neurons on and off to improve convergence

    model.add(tf.keras.layers.Dropout(0.25))

    #flatten since too many dimensions, we only want a classification output

    model.add(tf.keras.layers.Flatten())

    #fully connected to get all relevant data

    model.add(tf.keras.layers.Dense(128, activation='relu'))

    #one more dropout for convergence' sake :) 

    model.add(tf.keras.layers.Dropout(0.5))

    #output a softmax to squash the matrix into output probabilities

    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    

    return model
#here we build the network, instance an optizmier and compile it

cnn = build_cnn(X_train.shape[1:])

optimizer = tf.keras.optimizers.Adam(lr=0.01)

cnn.compile(loss=tf.keras.losses.categorical_crossentropy,

              metrics=['accuracy'], optimizer=optimizer)
history = cnn.fit(X_train, Y_train, epochs=10, validation_split=0.1)
#plotting accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
evaluation = cnn.evaluate(X_val, Y_val, verbose=2)

print(evaluation)
pred_for_idx = cnn(X_train[idx:idx+1, :, :, :])
idx = 1026

plt.imshow(X_train[idx, :, :, 0], cmap='gist_gray')

print("Predicted label of the test sample {}: {}".format(idx, np.argmax(pred_for_idx[0]), axis=-1))
noise = np.random.random((28,28,1))

pred_for_noise = cnn(np.array([noise]))

plt.imshow(noise[:, :, 0], cmap='gist_gray')

print("Predicted label of the test sample {}: {}".format(idx, np.argmax(pred_for_noise[0]), axis=-1))

def build_bayesian_bcnn_model(input_shape):

    

    """

    Here we use tf.keras.Model to use our graph as a Neural Network:

    We select our input node as the net input, and the last node as our output (predict node).

    Note that our model won't be compiled, as we are usign TF2.0 and will optimize it with

    a custom @tf.function for loss and a @tf.function for train_step

    Our input parameter is just the input shape, a tuple, for the input layer

    """

    

    model_in = tf.keras.layers.Input(shape=input_shape)

    conv_1 = tfp.python.layers.Convolution2DFlipout(32, kernel_size=(3, 3), padding="same", strides=2)

    x = conv_1(model_in)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation('relu')(x)

    conv_2 = tfp.python.layers.Convolution2DFlipout(64, kernel_size=(3, 3), padding="same", strides=2)

    x = conv_2(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Flatten()(x)

    dense_1 = tfp.python.layers.DenseFlipout(512, activation='relu')

    x = dense_1(x)

    dense_2 = tfp.python.layers.DenseFlipout(10, activation=None)

    model_out = dense_2(x)  # logits

    model = tf.keras.Model(model_in, model_out)

    return model
"""

this is our loss function: a sum of KL Divergence and Softmax crossentropy

We use the @tf.function annotation becuase of TF2.0, and need no placeholders

we get each loss and return its mean

"""



@tf.function

def elbo_loss(labels, logits):

    loss_en = tf.nn.softmax_cross_entropy_with_logits(labels, logits)

    loss_kl = tf.keras.losses.KLD(labels, logits)

    loss = tf.reduce_mean(tf.add(loss_en, loss_kl))

    return loss
"""

this is our train step with tf2.0, very ellegant:

We do our flow of the tensors over the model recording its gradientes

Then, our gradient tape to give us a list of the gradients of each parameter in relation of the loss

we dan ask our previously instanced optimizer to apply those gradients to the variable

It is cool to see that it works even with TensorFlow probability- probabilistic layers parameters

"""

@tf.function

def train_step(images, labels):

    with tf.GradientTape() as tape:

        logits = bcnn(X_train)

        loss = elbo_loss(labels, logits)

    gradients = tape.gradient(loss, bcnn.trainable_variables)

    optimizer.apply_gradients(zip(gradients, bcnn.trainable_variables))

    return loss



def accuracy(preds, labels):

    return np.mean(np.argmax(preds, axis=1) == np.argmax(labels, axis=1))
bcnn = build_bayesian_bcnn_model(X_train.shape[1:])

optimizer = tf.keras.optimizers.Adam(lr=0.01)
"""

in our train step we can see that it lasts more tha na normal CNN to converge

on the other side, we can have the confidence interval for our predictions, which are 

wonderful in terms of taking sensitive predictions

"""

times = []

accs = []

val_accs = []

losses = []

val_losses = []

for i in range(20):

    tic = time.time()

    loss = train_step(X_train, Y_train)

    preds = bcnn(X_train)

    acc = accuracy(preds, Y_train)

    accs.append(acc)

    losses.append(loss)

    

    val_preds = bcnn(X_val)

    val_loss = elbo_loss(Y_val, val_preds)

    val_acc = accuracy(Y_val, val_preds)

    

    val_accs.append(val_acc)

    val_losses.append(val_loss)

    tac = time.time()

    train_time = tac-tic

    times.append(train_time)

    

    print("Epoch: {}: loss = {:7.3f} , accuracy = {:7.3f}, val_loss = {:7.3f}, val_acc={:7.3f} time: {:7.3f}".format(i, loss, acc, val_loss, val_acc, train_time))
#plotting accuracy

plt.plot(np.array(accs), label="acc")

plt.plot(np.array(val_accs), label="val_acc")

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()



# Plot training & validation loss values

plt.plot(np.array(losses), label="loss")

plt.plot(np.array(val_losses), label="val_loss")

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
def plot_pred_hist(y_pred, n_class, n_mc_run, n_bins=30, med_prob_thres=0.2, n_subplot_rows=2, figsize=(25, 10)):

    bins = np.logspace(-n_bins, 0, n_bins+1)

    fig, ax = plt.subplots(n_subplot_rows, n_class // n_subplot_rows + 1, figsize=figsize)

    for i in range(n_subplot_rows):

        for j in range(n_class // n_subplot_rows + 1):

            idx = i * (n_class // n_subplot_rows + 1) + j

            if idx < n_class:

                ax[i, j].hist(y_pred[idx], bins)

                ax[i, j].set_xscale('log')

                ax[i, j].set_ylim([0, n_mc_run])

                ax[i, j].title.set_text("{} (median probability: {:.2f}) ({})".format(str(idx),

                                                                               np.median(y_pred[idx]),

                                                                               str(np.median(y_pred[idx]) >= med_prob_thres)))

            else:

                ax[i, j].axis('off')

    plt.show()
n_mc_run = 50

med_prob_thres = 0.35



y_pred_logits_list = [bcnn(X_val) for _ in range(n_mc_run)]  # a list of predicted logits

y_pred_prob_all = np.concatenate([tf.nn.softmax(y, axis=-1)[:, :, np.newaxis] for y in y_pred_logits_list], axis=-1)

y_pred = [[int(np.median(y) >= med_prob_thres) for y in y_pred_prob] for y_pred_prob in y_pred_prob_all]

y_pred = np.array(y_pred)



idx_valid = [any(y) for y in y_pred]

print('Number of recognizable samples:', sum(idx_valid))



idx_invalid = [not any(y) for y in y_pred]

print('Unrecognizable samples:', np.where(idx_invalid)[0])



print('Test accuracy on MNIST (recognizable samples):',

      sum(np.equal(np.argmax(Y_val[idx_valid], axis=-1), np.argmax(y_pred[idx_valid], axis=-1))) / len(Y_val[idx_valid]))



print('Test accuracy on MNIST (unrecognizable samples):',

      sum(np.equal(np.argmax(Y_val[idx_invalid], axis=-1), np.argmax(y_pred[idx_invalid], axis=-1))) / len(Y_val[idx_invalid]))
class_nmr = 10

plt.imshow(X_val[0, :, :, 0], cmap='gist_gray')

print("True label of the test sample {}: {}".format(0, np.argmax(Y_val[0], axis=-1)))



plot_pred_hist(y_pred_prob_all[0], class_nmr, n_mc_run, med_prob_thres=med_prob_thres)
class_nmr = 10

invalids = 0

for idx in np.where(idx_invalid)[0]:

    plt.imshow(X_val[idx, :, :, 0], cmap='gist_gray')

    print("True label of the test sample {}: {}".format(idx, np.argmax(Y_val[idx], axis=-1)))



    plot_pred_hist(y_pred_prob_all[idx], class_nmr, n_mc_run, med_prob_thres=med_prob_thres)



    if any(y_pred[idx]):

        print("Predicted label of the test sample {}: {}".format(idx, np.argmax(y_pred[idx], axis=-1)))

    else:

        print("I don't know!")

    invalids += 1

    if invalids > 5:

        break
end = time.time()

runtime = (end - start) / 60

print("This notebook ran in {:7.3f} minutes".format(runtime))