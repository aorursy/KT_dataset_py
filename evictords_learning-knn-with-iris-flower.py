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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from matplotlib.colors import ListedColormap

from matplotlib.colors import ListedColormap

print('extra libs to use')
#loading the data set

df = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')
#check the columns

df.columns
#check some data

df.head(5)
#how to count the values of a column

df['species'].value_counts()
#a way to selecting columns with pandas

def selectColumnsForX(dataframe):

    '''From the iris dataframe select the clomuns to be used in the KNN machine learning algorithm

        columns in use = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    '''

    return dataframe[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]



X_columns = selectColumnsForX(df)
#using loc to select X as values and array type

df.loc[:, ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
#using iloc to select columns and rows

df.iloc[:,:4].values
#building the data set for KNN use

x = df.iloc[:,:4].values

y = df.iloc[:,4].values
#creating the test and train matrix

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
#using scaling for the matrix

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
#training the algorithm

classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)

classifier.fit(x_train, y_train)
#using KNN to predict

y_pred = classifier.predict(x_test)
#building confuse matrix

confusion_matrix(y_test, y_pred)
score = 0



for i in range(len(y_pred)):

    if y_test[i] == y_pred[i]:

        score += 1



final = round((score/len(y_pred)) * 100, 2)

print('Hit: '+str(final)+'%')