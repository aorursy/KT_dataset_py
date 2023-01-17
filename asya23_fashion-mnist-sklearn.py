# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score



mnist_train = pd.read_csv("../input/fashion-mnist_train.csv") # грузим csv

mnist_test = pd.read_csv("../input/fashion-mnist_test.csv")

train_data=mnist_train.values[:, 1:] #убираем столбец отвественный за labels

test_data=mnist_test.values[:, 1:] 

train_labels = mnist_train.values[:,0] #сохраняем все значения одежды

test_labels = mnist_test.values[:,0] 

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] # классы одежды
for i in range(25): # красиво вывели и посмотрели первые 25 элементов

    plt.subplot(5,5,i+1)

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    plt.xticks([])

    plt.yticks([])

    image_row=mnist_train.values[i, 1:]

    plt.imshow(image_row.reshape(28,28), cmap="Greys")

    plt.xlabel(class_names[train_labels[i]])
mlp_classifier = MLPClassifier(verbose=True)

mlp_classifier = mlp_classifier.fit(train_data, train_labels) #учим модель

mlp_predictions = mlp_classifier.predict(test_data) #на тестовых данных запускаем

print(accuracy_score(test_labels, mlp_predictions)*100)
test_id =4568 #простая проверка

plt.imshow(test_data[test_id, :].reshape(28,28), cmap="Greys")

print(class_names[test_labels[test_id]])

class_names[int(mlp_classifier.predict(test_data[test_id,:].reshape(1,784)))]
mlp_classifier.get_params().keys()
parameters_grids = {

    'activation' : ['identity', 'logistic', 'tanh', 'relu'],

    'solver' : ['lbfgs', 'sgd', 'adam'],

    #'learning_rate_init': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],

   # 'alpha': np.linspace(0.0001,0.001, num=5),

   # 'hidden_layer_sizes': [(50,),(100,),(150,),(200,),(250,),(300,),(350,),(400,),(450,),(500,)]

}
grid_model = GridSearchCV(mlp_classifier, parameters_grids, scoring = 'accuracy' ,verbose=10)
grid_model.best_estimator_