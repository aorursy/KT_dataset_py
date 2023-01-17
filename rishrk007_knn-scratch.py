# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn import datasets
iris=datasets.load_iris()
iris
x=iris.data

y=iris.target
y
x
import random
from scipy.spatial import distance
def euc(a,b):

    return distance.euclidean(a,b)
class scapKNN():

    def fit(self,x_train,y_train):

        self.x_train=x_train

        self.y_train=y_train

        

    def predict(self,x_test):

        pred=[]

        for row in x_test:

            label=self.closest(row)

            pred.append(label)

        return pred    

    

    def closest(self,row):

        best_dis=euc(row,self.x_train[0])

        best_index=0

        for i in range(1,len(self.x_train)):

            dis=euc(row,self.x_train[i])

            if(dis<best_dis):

                best_dis=dis

                best_index=i

        return self.y_train[best_index]

                
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5)
my_classifier=scapKNN()
my_classifier.fit(x_train,y_train)
prediction=my_classifier.predict(x_test)
prediction
from sklearn.metrics import accuracy_score
print(accuracy_score(prediction,y_test))