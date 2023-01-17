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
data = pd.read_csv('../input/wisconsin_breast_cancer.csv') #read the values into variable 'data'

data = data.fillna(0) #Replace all NaN values with 0.0

data.head() #Show the first 5 rows of the dataset
data.info() #Displays key information about the data type and the columns within the data
import seaborn as sns

#sns.pairplot(data=data, palette='Set2')

sns.pairplot(data, hue='class') #pairplot that is not partioned by class (benign or malignant)
from sklearn.model_selection import train_test_split



#iloc function is an array defined by [rows, columns]



x = data.iloc[:, 1:10] #set x to all columns except ID and Class

y = data.iloc[:, -1] #set y to the Class column



#train_split_test the x and y values with a test size of 20%

x_train, x_test, y_train,y_test = train_test_split(x, y, test_size=0.2)



#print to see if the results are approximately what they should be

print(x_train.shape, y_train.shape) 
from sklearn.svm import SVC



model = SVC()

model.fit(x_train, y_train)
#print( y_train[ :5]) debugging to see if the first 5 values of the y_train (debugging)
pred = model.predict(x_test)



#both the prints should show same numbers in same order

print(pred[:10])

print(y_test[:10])
from sklearn.metrics import confusion_matrix, classification_report





#confusion matrix is used to show the performance of the test model compared to the true values of the dataset

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
from sklearn.model_selection import train_test_split



#iloc function is an array defined by [rows, columns]



xsub = data.iloc[:, [1,2,3,7]] #set x to all our promising features

y = data.iloc[:, -1] #set y to the Class column



#train_split_test the subx and y values with a test size of 20%

x_subtrain, x_subtest, y_train,y_test = train_test_split(xsub, y, test_size=0.2)



#print to see if the results are approximately what they should be

print(x_subtrain.shape, y_train.shape) 
from sklearn.svm import SVC



submodel = SVC()

submodel.fit(x_subtrain, y_train)
subpred = submodel.predict(x_subtest)



#both the prints should show same numbers in same order

print(subpred[:5])

print(y_test[:5])
from sklearn.metrics import confusion_matrix, classification_report





#confusion matrix is used to show the performance of the test model compared to the true values of the dataset

print(confusion_matrix(y_test, subpred))
print(classification_report(y_test, subpred))
x = data.iloc[:, 1:10] #set x to all columns except ID and Class

y = data.iloc[:, -1] #set y to the Class column



#train_split_test the x and y values with a test size of 20%

x_train, x_test, y_train,y_test = train_test_split(x, y, test_size=0.2)



#print to see if the results are approximately what they should be

print(x_train.shape, y_train.shape)
 



from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors = 1)

neigh.fit(x_train, y_train)
predn = neigh.predict(x_test)



#both the prints should show same numbers in same order

print(predn[:10])

print(y_test[:10])
print(confusion_matrix(y_test,predn))

print(classification_report(y_test, predn))
x = data.iloc[:, 1:10] #set x to all columns except ID and Class

y = data.iloc[:, -1] #set y to the Class column



#train_split_test the x and y values with a test size of 20%

x_train, x_test, y_train,y_test = train_test_split(x, y, test_size=0.2)



#print to see if the results are approximately what they should be

print(x_train.shape, y_train.shape)
from sklearn.neighbors import KNeighborsClassifier

neigh2 = KNeighborsClassifier(n_neighbors = 5)

neigh2.fit(x_train, y_train)
predn2 = neigh2.predict(x_test)



#both the prints should show same numbers in same order

print(predn2[:10])

print(y_test[:10])
print(confusion_matrix(y_test,predn2))
print(classification_report(y_test, predn2))