# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
bilgiler = pd.read_csv('/kaggle/input/banknot-pattern-recognizer/data_banknote_authentication.txt' , encoding='latin-1')
bilgiler
def weight_graph(w0array, w1array,w2array,w3array, number_of_weights_to_graph=100):

    epochs = len(w0array)

    # epoch değerleri

    num_per_epoch = epochs/number_of_weights_to_graph



    w0_to_graph = []

    w1_to_graph = []

    w2_to_graph = []

    w3_to_graph = []

    

    epoch_to_graph = []

    # ağırlıklara göre graph atma

    for i in range(number_of_weights_to_graph):

        epoch_to_graph.append(int(num_per_epoch*i))

        w0_to_graph.append(w0array[int(num_per_epoch*i)])

        w1_to_graph.append(w1array[int(num_per_epoch*i)])

        w2_to_graph.append(w2array[int(num_per_epoch*i)])

        w3_to_graph.append(w3array[int(num_per_epoch*i)])

    plt.plot(epoch_to_graph, w0_to_graph, 'r',epoch_to_graph, w1_to_graph,'b')

    plt.show()
def train_svm(X, Y, epochs=100, learning_rate=1):

    # 3 veri için 

    w = np.zeros(len(X[0]))



    w0_per_epoch = []

    w1_per_epoch = []

    w2_per_epoch = []

    w3_per_epoch = []



    # Training

    print("starts training")

    for epoch in range(1, epochs):

        for i, x in enumerate(X):

            # It there is an error

            if (Y[i] * np.dot(X[i], w)) < 1:

                w = w + learning_rate * ((X[i] * Y[i]) + (-2 * (1/epochs) * w))

            else:

                w = w + learning_rate * (-2 * (1/epochs) * w)

        w0_per_epoch.append(w[0])

        w1_per_epoch.append(w[1])

        w2_per_epoch.append(w[2])

        w3_per_epoch.append(w[3])



    weight_graph(w0_per_epoch, w1_per_epoch,w2_per_epoch,w3_per_epoch)

    return w
def predict(X, w):

    Y = np.dot(X, w)

    return Y
def show_svm_graph(X, Y, w):

   

    for i in range(len(X)):

        #Kural 1 den farklı değil olarak aldık

        if Y[i] == 1:

            plt.scatter(X[i][0], X[i][1], s=120, marker='_', linewidths=2)

        else:

            plt.scatter(X[i][0], X[i][1], s=120, marker='+', linewidths=2)



    # SGD errorü

    x2=[w[0]*0.65,w[1],-w[1],w[0]]

    x3=[w[0]*0.65,w[1],w[1],-w[0]]



    x2x3 =np.array([x2,x3])

    X,Y,U,V = zip(*x2x3)

    ax = plt.gca()

    ax.quiver(X,Y,U,V,scale=1, color='blue')

    plt.show()
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

X = bilgiler.drop('0', axis=1)

y = bilgiler['0']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

xArray = np.asarray(X_train)

yArray = np.asarray(y_train)

w = train_svm(xArray, yArray)

show_svm_graph(xArray,yArray, w)

# 100 epocha göre çizimi 

# daha yüksek epoch da yanlış yerleşmekte çizgi