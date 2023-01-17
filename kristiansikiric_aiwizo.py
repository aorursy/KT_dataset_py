import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv("/kaggle/input/mnist-in-csv/mnist_train.csv")

test = pd.read_csv("/kaggle/input/mnist-in-csv/mnist_test.csv")



x = train.drop(['label'], inplace = False, axis = 1)

y = train['label']



test_x = test.drop(['label'], inplace = False, axis = 1)

test_y = test['label']

indices = []



for i in range(10):

    indices.append(y.loc[y == i].index[0])

    

train_x = x.loc[indices]

train_y = y.loc[indices]

unlabeled_x = pd.DataFrame.copy(x)

unlabeled_x.drop(indices,inplace = True)
from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier



svm = SGDClassifier()



while(len(unlabeled_x)):

    svm_fit = svm.fit(train_x,train_y)

    

    population = int(len(train_x))

    if population > len(unlabeled_x):

        population = len(unlabeled_x)

    sample = unlabeled_x.sample(n = population,replace = False)

    unlabeled_x.drop(sample.index,inplace = True)

    

    psuedo_labels = svm_fit.predict(sample)

    train_x = pd.concat([train_x,sample])

    train_y = pd.concat([train_y,pd.DataFrame(psuedo_labels)])[0]



svm_fit = svm.fit(train_x,train_y)

svm_fit.score(test_x,test_y)