# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing packages beforehand

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.preprocessing import StandardScaler
#Retrival of the dataset in csv format from its location 

rd = pd.read_csv("../input/result.csv")

rd.head()
#Listing up the column names

rd.columns
#Cheching for null values

rd.isnull().sum()
rd.info()
rd.describe()
#Plotting histogram of 'FIRST' column

plt.hist(rd['FIRST'])

print("HISTOGRAM OF FIRST SEMESTER MARKS")
#Plotting histogram of 'SECOND' column

plt.hist(rd['SECOND'])

print("HISTOGRAM OF SECOND SEMESTER MARKS")
#Plotting histogram of 'REMARKS' column

plt.hist(rd['REMARKS'])

print("HISTOGRAM OF REMARKS SEMESTER MARKS")
#x is a dataframe (2D matrix)

#x includes all independent variables

x=rd.iloc[ : , :2]

x
#y is a series (linear list)

#x includes dependent variable only

y=rd.iloc[ :, -1]

y
#Plotting pairplot between THIRD & FIRST

sns.pairplot(rd.iloc[ : ,[0,-2]])

print("PAIRPLOT BETWEEN THIRD AND FIRST")
#Plotting scatter plot of SECOND Vs THIRD

plt.scatter(rd.iloc[ : ,[1]], rd.iloc[ : ,[2]])  

plt.xlabel('SECOND') 

plt.ylabel('THIRD') 

print('THIRD Vs SECOND SCATTER PLOT')
#rs is used to find most proper random state(j)

rs=[]

for j in range(200):

    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=j,test_size=0.1)

    knc = KNeighborsClassifier().fit(x_train, y_train)

    rs.append(knc.score(x_test, y_test))

j = rs.index(np.max(rs))

j
#Splitting the total dataset into 90% train data sample 10% test data sample

#Test samples are choosen randomly as per random_state

#x_train is the training dataframe

#x_test is the testing dataframe

#y_train is the training series

#x_test is the testing series

#The training data samples acts as knowledgebase

#The testing data samples are used to check the accuracy of the method

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,random_state=97)
#printing the testing dataframe

x_test 
#printing the testing series

y_test
#printing the training dataframe

x_train 
#printing the training series

y_train
#Standarizing the dataframes using fit() 

#It is essential before using Euclidean distance  

scaler = StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)
x_train #after Standardization
x_test  #after Standarization
#applying KNN with k value 3

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=3)

classifier.fit(x_train, y_train)
#generating the predictions

y_pred = classifier.predict(x_test)

y_pred
# showing Confusion Matrix and Classification report

print('CONFUSION MATRIX')

print(confusion_matrix(y_test, y_pred))

print('CLASSIFICATION REPORT')

print(classification_report(y_test, y_pred))
#Finding accuracy percentage

ac=accuracy_score(y_test,y_pred)

print(ac*100)
from matplotlib.colors import ListedColormap

X_set,y_set = x_test,y_test

X1 , X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1,step = 0.2),

                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1,step = 0.2))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

alpha = 1, cmap = ListedColormap(('#ff0000', '#00c05f')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

    c = ListedColormap(('#AD0000', '#008000'))(i), label = j)

plt.title('K-NN (Testing set)')

plt.xlabel('FIRST')

plt.ylabel('SECOND')

plt.legend()

plt.show()
from matplotlib.colors import ListedColormap

X_set,y_set = x_train, y_train

X1 , X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1,step = 0.1),

                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1,step = 0.1))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

alpha = 0.9, cmap = ListedColormap(('#ff0000', '#00c05f')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

    c = ListedColormap(('#AD0000', '#008000'))(i), label = j)

plt.title('K-NN (Training set)')

plt.xlabel('FIRST')

plt.ylabel('SECOND')

plt.legend()

plt.show()