# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
clas = pd.read_csv('../input/titanic/data_cleaned.csv')

clas.head()
clas.shape
clas.info()
x = clas.drop(['Survived'],axis=1)

y = clas['Survived']

x.shape,y.shape
#Importing MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaled = scaler.fit_transform(x)
#Importing train_test_split function

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=12,stratify=y)

x_train.shape,x_test.shape,y_train.shape,y_test.shape
#Importing the classifier and accuracy metrics

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.metrics import f1_score
#Creating instance of KNN

ins = KNN(n_neighbors=9)



#Fit model

ins.fit(x_train,y_train)



#Predict model over test data

predict= ins.predict(x_test)

f1 = f1_score(predict,y_test)

print("The accuracy of f1_score = ",f1)
def elbow(K):

    #initialize an empty list

    error=[]

    

    #train model for every K

    for i in K:

        #Instance of KNN

        ins = KNN(n_neighbors=i)

        ins.fit(x_train,y_train)

        

        #Appending the f1_scores to the empty list

        predict= ins.predict(x_test)

        f1 = f1_score(predict,y_test)

        z = 1-f1

        error.append(z)

        

    return error
#Define the range of K after intervals

k = range(5,30,2)
#Calling the elbow function

test = elbow(k)
plt.plot(k,test)

plt.title("Elbow curve")

plt.xlabel("K")

plt.ylabel("Test Error")
#Creating instance of KNN

ins = KNN(n_neighbors = 5)



#Fit model

ins.fit(x_train,y_train)



#Predict model over test data

predict= ins.predict(x_test)

f1 = f1_score(predict,y_test)

print("Test f1_score = ",f1)
reg = pd.read_csv('../input/titanic/train_cleaned.csv')

reg.head()
reg.shape
reg.info()
X = reg.drop(['Item_Outlet_Sales'],axis=1)

Y = reg['Item_Outlet_Sales']

X.shape,Y.shape
scale = MinMaxScaler()

x_scaled = scale.fit_transform(X)
X = pd.DataFrame(x_scaled)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=32)

X_train.shape,X_test.shape,Y_train.shape,Y_test.shape
#Importing the regressor and accuracy metrics

from sklearn.neighbors import KNeighborsRegressor as KNN

from sklearn.metrics import mean_squared_error as mse

#Creating instance of KNN

s = KNN(n_neighbors=7)



#Fit model

s.fit(X_train,Y_train)



#Predict model over test data

pred= s.predict(X_test)

er = mse(pred,Y_test)

print("The accuracy of MSE score = ",er)
def elbow(p):

    #initialize an empty list

    errornew=[]

    

    #train model for every p

    for i in p:

        #Instance of KNN

        s = KNN(n_neighbors=i)

        s.fit(X_train,Y_train)

        

        #Appending the f1_scores to the empty list

        pred= s.predict(X_test)

        mean = mse(pred,Y_test)

        mseer = 1-mean

        errornew.append(mseer)

        

    return errornew
#Define the range of K after intervals

p = range(5,20,2)
#Calling the elbow function

testnew = elbow(p)
#Plotting the curves

plt.plot(p,testnew)

plt.title("Elbow curve")

plt.xlabel("K")

plt.ylabel("Test Error")
#Creating instance of KNN

s = KNN(n_neighbors=20)



#Fit model

s.fit(X_train,Y_train)



#Predict model over test data

pred= s.predict(X_test)

er = mse(pred,Y_test)

print("The MSE score = ",er)