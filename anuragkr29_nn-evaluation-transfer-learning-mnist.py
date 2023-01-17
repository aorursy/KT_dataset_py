# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#!git clone https://github.com/yenchenlin/nn_robust_attacks.git

# cd nn_robust_attacks/
from setup_mnist import MNIST

import matplotlib.pyplot as plt

import time

import datetime as dt



# Import datasets, classifiers and performance metrics

from sklearn import datasets, svm, metrics

data = MNIST()

images = data.train_data

targets = data.train_labels
X_train_svm = images.flatten().reshape(55000, 784)

Y_train = np.argmax(targets,axis=1)
X_train_svm_small = X_train_svm[:1000]

Y_train_small = Y_train[:1000]
# Svm training



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train_svm_small, Y_train_small, test_size=0.1, random_state=42)





################ Classifier with good params ###########

# Create a classifier: a support vector classifier



param_C = 5

param_gamma = 0.05

classifier_svm = svm.SVC(C=param_C,gamma=param_gamma,verbose=True)



#We learn the digits on train part

start_time = dt.datetime.now()

print('Start learning at {}'.format(str(start_time)))

classifier_svm.fit(X_train, y_train)

end_time = dt.datetime.now() 

print('Stop learning {}'.format(str(end_time)))

elapsed_time= end_time - start_time

predicted = classifier_svm.predict(X_test)

expected = y_test

print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))
# utilities for RNN

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.layers.recurrent import SimpleRNN, LSTM, GRU 

from keras.utils import np_utils

from keras import backend as K



from distutils.version import LooseVersion as LV

from keras import __version__



from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot
# training RNN



nb_units = 50

img_rows, img_cols = 28, 28

classifier_RNN = Sequential()

nb_classes = 10

# Recurrent layers supported: SimpleRNN, LSTM, GRU:

classifier_RNN.add(SimpleRNN(nb_units,

                    input_shape=(img_rows, img_cols)))



# To stack multiple RNN layers, all RNN layers except the last one need

# to have "return_sequences=True".  An example of using two RNN layers:

#model.add(SimpleRNN(16,

#                    input_shape=(img_rows, img_cols),

#                    return_sequences=True))

#model.add(SimpleRNN(32))



classifier_RNN.add(Dense(units=nb_classes))

classifier_RNN.add(Activation('softmax'))



classifier_RNN.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

epochs = 10

X_train_RNN = images.flatten().reshape(55000, 28,28)

Y_train_RNN = targets

classifier_RNN.fit(X_train_RNN, 

                    Y_train_RNN, 

                    epochs=epochs, 

                    batch_size=128)

from joblib import dump, load

dump(classifier_RNN, 'RNN_model.h5') 
#model1.predict_classes(adv[9].flatten().reshape(1,28,28))
# Testing attack

import tensorflow as tf

import numpy as np

import time



from setup_cifar import CIFAR, CIFARModel

from setup_mnist import MNIST, MNISTModel

from setup_inception import ImageNet, InceptionModel



from l2_attack import CarliniL2

from l0_attack import CarliniL0

from li_attack import CarliniLi





def show(img):

    """

    Show MNSIT digits in the console.

    """

    remap = "  .*#"+"#"*100

    img = (img.flatten()+.5)*3

    if len(img) != 784: return

    print("START")

    for i in range(28):

        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))





def generate_data(data, samples, targeted=True, start=0, inception=False):

    """

    Generate the input data to the attack algorithm.

    data: the images to attack

    samples: number of samples to use

    targeted: if true, construct targeted attacks, otherwise untargeted attacks

    start: offset into data to use

    inception: if targeted and inception, randomly sample 100 targets intead of 1000

    """

    inputs = []

    targets = []

    for i in range(samples):

        if targeted:

            if inception:

                seq = random.sample(range(1,1001), 10)

            else:

                seq = range(data.test_labels.shape[1])



            for j in seq:

                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):

                    continue

                inputs.append(data.test_data[start+i])

                targets.append(np.eye(data.test_labels.shape[1])[j])

        else:

            inputs.append(data.test_data[start+i])

            targets.append(data.test_labels[start+i])



    inputs = np.array(inputs)

    targets = np.array(targets)



    return inputs, targets
#!wget https://raw.githubusercontent.com/codekansas/keras-resnet/master/resnet.py

!ls
# random forest code

from sklearn.ensemble import RandomForestClassifier



#digits = datasets.load_digits()

#print(digits.data)



classifier_RF = RandomForestClassifier(n_estimators=100, n_jobs=10,)

classifier_RF.fit(X_train,y_train)



# with open('MNIST_RFC.pickle','wb') as f:

# 	pickle.dump(clf, f)



# pickle_in = open('MNIST_RFC.pickle','rb')

# clf = pickle.load(pickle_in)



acc = classifier_RF.score(X_test,y_test)

print('RFC Score: ',acc)
classifier_RF.predict(np.reshape(np.array(inputs[0]),(1,-1)))
## train_models.py -- train the neural network models for attacking

##

## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.

##

## This program is licenced under the BSD 2-Clause licence,

## contained in the LICENCE file in this directory.



from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import SGD

import os

import tensorflow as tf



def train(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1, init=None):

    """

    Standard neural network training procedure.

    """

    model = Sequential()



    print(data.train_data.shape)

    

    model.add(Conv2D(params[0], (3, 3),

                            input_shape=data.train_data.shape[1:]))

    model.add(Activation('relu'))

    model.add(Conv2D(params[1], (3, 3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(params[2], (3, 3)))

    model.add(Activation('relu'))

    model.add(Conv2D(params[3], (3, 3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

    model.add(Dense(params[4]))

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(params[5]))

    model.add(Activation('relu'))

    model.add(Dense(10))

    

    if init != None:

        model.load_weights(init)



    def fn(correct, predicted):

        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,

                                                       logits=predicted/train_temp)



    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    

    model.compile(loss=fn,

                  optimizer=sgd,

                  metrics=['accuracy'])

    

    model.fit(data.train_data, data.train_labels,

              batch_size=batch_size,

              validation_data=(data.validation_data, data.validation_labels),

              nb_epoch=num_epochs,

              shuffle=True)

    



    if file_name != None:

        model.save(file_name)



    return model



def train_distillation(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1):

    """

    Train a network using defensive distillation.



    Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks

    Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami

    IEEE S&P, 2016.

    """

    if not os.path.exists(file_name+"_init"):

        # Train for one epoch to get a good starting point.

        train(data, file_name+"_init", params, 1, batch_size)

    

    # now train the teacher at the given temperature

    teacher = train(data, file_name+"_teacher", params, num_epochs, batch_size, train_temp,

                    init=file_name+"_init")



    # evaluate the labels at temperature t

    predicted = teacher.predict(data.train_data)

    with tf.Session() as sess:

        y = sess.run(tf.nn.softmax(predicted/train_temp))

        print(y)

        data.train_labels = y



    # train the student model at temperature t

    student = train(data, file_name, params, num_epochs, batch_size, train_temp,

                    init=file_name+"_init")



    # and finally we predict at temperature 1

    predicted = teacher.predict(data.train_data)



    print(predicted)

    

if not os.path.isdir('models'):

    os.makedirs('models')



train(MNIST(), "models/mnist", [32, 32, 64, 64, 200, 200], num_epochs=50)



train_distillation(MNIST(), "models/mnist-distilled-100", [32, 32, 64, 64, 200, 200],

                   num_epochs=50, train_temp=100)

with tf.Session() as sess:

    data, model =  MNIST(), MNISTModel("models/mnist", sess)

    rnn_model = load('RNN_model.h5')

    attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0)

    #attack = CarliniL0(sess, model, max_iterations=1000, initial_const=10,

    #                   largest_const=15)



    inputs, targets = generate_data(data, samples=2, targeted=True,

                                    start=0, inception=False)

    timestart = time.time()

    adv = attack.attack(inputs, targets)

    timeend = time.time()



    print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")



    for i in range(len(adv)):

#         print("Valid:")

#         show(inputs[i])

#         print("Adversarial:")

#         show(adv[i])

        print('-----------------------------------------------------------------------')

        print("Classification orig:", np.argmax(model.model.predict(inputs[i:i+1])))

        print("Classification:", np.argmax(model.model.predict(adv[i:i+1])))

        print("SVM_Classification orig : " , classifier_svm.predict(np.reshape(np.array(inputs[i]),(1,-1))))

        print("SVM_Classification : " , classifier_svm.predict(np.reshape(np.array(adv[i]),(1,-1))))

        print("RF_Classification orig : " , classifier_RF.predict(np.reshape(np.array(inputs[i]),(1,-1))))

        print("RF_Classification : " , classifier_RF.predict(np.reshape(np.array(adv[i]),(1,-1))))

        print("LSTM_Classification : ", rnn_model.predict_classes(inputs[i].flatten().reshape(1,28,28)))

        print("LSTM_Classification : ", rnn_model.predict_classes(adv[i].flatten().reshape(1,28,28)))

        print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)

        print('-----------------------------------------------------------------------')
classifier_RNN.predict(adv[0].flatten().reshape(1,28,28))
adv[i].shape
X_train_RNN[0].shape
g = [ 1.3699718 ,1.3495499,13.239627 ,  -3.4946742 ,  0.34508333 ,-5.2202,

  -7.7000027 , 13.237005 ,  -3.8523767 , -2.8774788 ]
np.argmax(g[0])