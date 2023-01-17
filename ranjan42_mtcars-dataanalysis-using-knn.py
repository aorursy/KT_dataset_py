#Importing the necessary libraries using Python
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
from sklearn.neighbors import KNeighborsClassifier
# Using the KNN algorithm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import preprocessing 
#Reading the data
cars = pd.read_csv('../input/mtcars.csv')
#Checking the dataset 
cars.head()
#checking for missing values
cars.isnull().sum()
X=cars.loc[:,['mpg','wt','hp','gear','cyl']]
y=cars.loc[:,'am']
X=preprocessing.scale(X)
#Splitting the data into Train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)
#checking train data set size
X_train.shape
model_knn =KNeighborsClassifier(n_neighbors=1)
#Training the model
model_knn.fit(X_train,y_train)
#Doing the prediction
y_predict = model_knn.predict(X_test)
#checking the model accuracy
confusion_matrix(y_test,y_predict)
print(classification_report(y_test,y_predict))
