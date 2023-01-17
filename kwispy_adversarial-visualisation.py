from sklearn import datasets
from sklearn.manifold import TSNE
import keras
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import load_model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Activation, Dense
import keras

import logging
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_tf import model_eval, model_argmax
from cleverhans.train import train
from cleverhans_tutorials.tutorial_models import ModelBasicCNN
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.attacks import BasicIterativeMethod

import matplotlib.pyplot as plt
from cleverhans.dataset import MNIST
#import plotly.plotly 
#import plotly.graph_objs as go
from keras.datasets import mnist
import foolbox
from cleverhans.utils_keras import KerasModelWrapper


def preProcessing(X_,y_):
    X_ = X_.reshape(len(X_), 784).astype('float32')
    #Xtest = Xtest.reshape(len(Xtest), 784).astype('float32')
    X_ = X_ / 255
    y_ = keras.utils.to_categorical(y_)
    return X_, y_


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(784, input_dim=784, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



def baseline_model_2D():
    # create model
    model = Sequential()
    model.add(Dense(100, input_dim=2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20,  activation='tanh'))
    model.add(Dense(10,  activation='tanh'))


    model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_trainF, y_trainF= preProcessing(X_train, y_train)
X_testF, y_testF= preProcessing(X_test, y_test)
X_train.shape
#from sklearn.decomposition import PCA
#pca= PCA(n_components=2)
#X_2d = pca.fit_transform(X_trainF[:2000].reshape(2000,784))
#X_2d.shape
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X_trainF[:2000].reshape(2000,784))
plt.figure(figsize=(6, 5))
plt.title("distribution t-sne on iteration from 0 to 10 with epsilon_iter 0.01")
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', '#ff66cc', 'orange', 'purple'
for i, c, label in zip(range(10), colors, range(10)):
    plt.scatter(X_2d[y_train[:2000] == i, 0], X_2d[y_train[:2000] == i, 1], c=c, label=label)
plt.legend()
plt.show()
model = baseline_model_2D()
# Fit the model
model.fit(X_2d[:1500], y_trainF[:1500], validation_data=(X_2d[1500:2000], y_trainF[1500:2000]), epochs=200, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_2d[1500:2000], y_trainF[1500:2000], verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
sscores = model.predict_classes(X_2d[1500:2000])
# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.1
    y=np.argmax(y, axis=1)
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.figure(figsize=(20,10))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.colorbar()
X_2d.shape
plot_decision_boundary(lambda ww: model.predict_classes(ww), X_2d[1500:2000], y_trainF[1500:2000])
plt.scatter(X_2d[1600,0], X_2d[1600,1], c='r' , marker='s',s=100 )

print(model.predict_classes(X_2d[1600].reshape(1,2)))
print(y_train[1600])
def get_bim_params(eps_iter = 0.01, nb_iter= 100):
    bim_params = {'eps_iter': eps_iter,
                  'nb_iter': nb_iter,
                }

    return bim_params


from sklearn.preprocessing import StandardScaler
ssc= StandardScaler()
X_2d_ssc=ssc.fit_transform(X_2d)
sess = tf.Session()

print("Created TensorFlow session.")
nb_classes = 10
x = tf.placeholder(tf.float32, shape=(None, 2 ))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))



modelc = KerasModelWrapper(baseline_model_2D())
softmax_name = modelc._get_softmax_name()
preds = modelc.get_logits(x)
print(softmax_name)
loss = CrossEntropy(modelc, smoothing=0.1)
print("Defined TensorFlow model graph.")


x_train, y_train = X_2d_ssc[:1500].reshape(1500,2), y_trainF[:1500]
x_test, y_test = X_2d_ssc[1500:2000].reshape(500,2), y_trainF[1500:2000]

###########################################################################
# Training the model using TensorFlow
###########################################################################
NB_EPOCHS = 200
BATCH_SIZE = 30
LEARNING_RATE = .001
SOURCE_SAMPLES = 10

# Train an MNIST model
train_params = {
  'nb_epochs': NB_EPOCHS,
  'batch_size': BATCH_SIZE,
  'learning_rate': LEARNING_RATE
}
sess.run(tf.global_variables_initializer())
rng = np.random.RandomState([2017, 8, 30])
train(sess, loss, x_train, y_train, args=train_params, rng=rng)

# Evaluate the accuracy of the MNIST model on legitimate test examples
eval_params = {'batch_size': BATCH_SIZE}
accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
print('Test accuracy on legitimate test examples: {0}'.format(accuracy))



x_test[900:1000]


bim_op = BasicIterativeMethod(modelc, sess=sess)
advs = bim_op.generate_np(x_test[:200], **get_bim_params(nb_iter=20, eps_iter=0.01))
pred_class_adv=model_argmax(sess, x, preds, advs)
pred_class_real=model_argmax(sess, x, preds, x_test[:200])
res_advs = np.array(pred_class_adv ==np.argmax(y_test[:200], axis=1)).astype(int)
accuracy_advs=res_advs.sum()/len(res_advs)
res_real = np.array(pred_class_real ==np.argmax(y_test[:200], axis=1)).astype(int)
accuracy_real=res_real.sum()/len(res_real)
print(accuracy_real)
print(accuracy_advs)
idx_advs= np.where(pred_class_real!=pred_class_adv)
idx_advs[0].shape
plt.figure(figsize=(20,20))
for i,idx in enumerate(idx_advs[0][:30]):
    plt.subplot(5,6,i+1)
    plt.title("real : {} , adv {} idx {}".format(pred_class_real[idx],pred_class_adv[idx], idx))
    plt.imshow(X_train[1500+idx].reshape(28,28))
plt.figure(figsize=(10,10))
plot_decision_boundary(lambda ww: model_argmax(sess, x, preds, ww), x_test.reshape(500,2), y_trainF[1500:2000])
for i in idx_advs[0][:30]:
    plt.scatter(x_test[i,0], x_test[i,1], c='black' ,marker="x" )
    plt.scatter(advs[i,0], advs[i,1], c='r' , marker="x")
    plt.annotate("idx {}".format(i), xy=(advs[i,0], advs[i,1]), xytext=(advs[i,0]+0.01, advs[i,1]+0.01),
            )
    plt.annotate("idx {}".format(i), xy=(x_test[i,0], x_test[i,1]), xytext=(x_test[i,0]+0.01, x_test[i,1]+0.01),
        
        )

