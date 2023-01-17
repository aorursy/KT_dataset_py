import h5py

import os

import tflearn



import numpy as np

import pandas as pd

import tensorflow as tf





from matplotlib import pyplot as plt

from sklearn.svm import LinearSVC
with h5py.File("../input/full_dataset_vectors.h5") as hf:

    X_train = hf["X_train"][:]

    y_train = hf["y_train"][:]

    X_test = hf["X_test"][:]

    y_test = hf["y_test"][:]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
clf = LinearSVC()

#clf.fit(X_train, y_train)
#print("Test Score: ", clf.score(X_test, y_test))
y_train = pd.get_dummies(y_train).values

y_test= pd.get_dummies(y_test).values
N_RUNS = 0
scores = []

predictions = np.zeros((X_test.shape[0], 10))



for i in range(N_RUNS):

    with tf.Graph().as_default():



        net = tflearn.input_data(shape=[None, 4096])

        net = tflearn.fully_connected(net, 128,

                                      activation='relu',

                                      weights_init='xavier',                                      

                                      regularizer='L2')

        net = tflearn.dropout(net, 0.5)

        net = tflearn.fully_connected(net, 128,

                                      activation='relu',

                                      weights_init='xavier',                                      

                                      regularizer='L2')

        net = tflearn.dropout(net, 0.5)

        net = tflearn.fully_connected(net, 10, activation='softmax')

        net = tflearn.regression(net)



        model = tflearn.DNN(net,

                            tensorboard_verbose=0,

                            best_checkpoint_path="best",

                            best_val_accuracy=0.64)



        model.fit(X_train,

                  y_train, 

                  validation_set=0.2,

                  show_metric=True,

                  n_epoch=20)

        

        best_name = "best"

        best_score = 0

        for cp in os.listdir():

            if cp.startswith("best"):

                score = int(cp.split(".")[0].split("t")[1])

                if score > best_score:

                    best_score = score

        best_name += str(best_score)

        

        if best_name != "best0":

            model.load(best_name)

            score = best_score / 10000

            scores.append(score)

            print("\n", "SCORE:", score, "\n\n")



            prediction = np.array(model.predict(X_test))

            predictions += prediction * score        

        

        for f in os.listdir():

            os.remove(f)