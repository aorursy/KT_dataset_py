# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#reading the data



fifa19_data=pd.read_csv('../input/data.csv')



#checking the shape and all columns



print(fifa19_data.shape)

print(fifa19_data.columns)



#visualizing the head of my dataset



fifa19_data.head()
#counting nulls



%time fifa19_data.isnull().sum()
#checking the tendencies of the data



%time fifa19_data.describe()

data=fifa19_data.iloc[:,0:18]

data
#counting null values 



%time data.isnull().sum()

#replacing null values



data["Club"].fillna("NA", inplace = True)



data["Preferred Foot"].fillna("Both", inplace=True)



data["Weak Foot"].fillna("NA", inplace=True)



data["Skill Moves"].fillna(1, inplace=True)



data["International Reputation"].fillna(1, inplace=True)
#relationship between the attributes 



%time sns.heatmap(data.corr(),annot=True)

plt.title("Correlation between all attributes")

plt.show()
from math import log

def value_to_int(df_value):

    try:

        value = float(df_value[1:-1])

        suffix = df_value[-1:]



        if suffix == 'M':

            value = value * 1000000

        elif suffix == 'K':

            value = value * 1000

    except ValueError:

        return 0

    return log(int(value))
#dropping unnamed 0 and ID columns as we don't have any use of them



data = data.drop(columns=['Unnamed: 0','ID'])



#cleaning value and wages to be integers and getting log values so as to have better correlation



data["Value"]=data["Value"].apply(value_to_int)

data["Wage"]=data["Wage"].apply(value_to_int)





data.shape
data.head()
#cleaning null values



data=data[data["Value"]!=0]

data=data[data["Wage"]!=0]
#plotting each pair of attributes



%time sns.pairplot(data)

plt.show()
#Prediction of potential only from overall

#Dividing the features

X=data[["Overall"]].values

y=data[["Potential"]].values



#splitting data into train test

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1 )



X_train.shape, y_train.shape



from sklearn.linear_model import LinearRegression



regr=LinearRegression()

regr.fit(X=X_train, y=y_train )

y_pred=regr.predict(X_train)



#intercept,score,RMSE

regr.intercept_, regr.score(X_train, y_train),np.sqrt(np.mean((y_train-y_pred)**2))
#predicting test data

y=regr.predict(X_test)



#score, RMSE

print(regr.score(X_test,y_test))

print("Mean Squared error:{}".format(np.sqrt(np.mean((y_test-y)**2))))

      

#Visualizing the prediction

    

plt.scatter(x=X_test,y=y_test, s=15)

plt.plot(X_test,regr.predict(X_test))

plt.xlabel("Overall")

plt.ylabel("Potential")

plt.show()
#Prediction of potential only from overall and age of the players

#Dividing the features

X1=data[["Overall","Age"]].values

y1=data[["Potential"]].values



#splitting data into train test

from sklearn.model_selection import train_test_split



X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.1 )



print(X1_train.shape, y1_train.shape)



from sklearn.linear_model import LinearRegression



regr1=LinearRegression()

regr1.fit(X=X1_train, y=y1_train )

y1_pred=regr1.predict(X1_train)



#intercept,score,RMSE

regr1.intercept_, regr1.score(X1_train, y1_train),np.sqrt(np.mean((y1_train-y1_pred)**2))


y1=regr1.predict(X1_test)

print(regr1.score(X1_test,y1_test))

print("MSE:{}".format(np.sqrt(np.mean((y1_test-y1)**2))))



#seperating overall and age

ovr=X1_test[:,0].reshape(X1_test.shape[0],1)

age=X1_test[:,1].reshape(X1_test.shape[0],1)



#3d plotting the predicted outcome

fig=plt.figure()

ax=fig.add_subplot(111, projection='3d')

fig.set_size_inches(16,9)

ax.scatter(xs=ovr, ys=age, zs=y1_test)

ax.plot(xs=ovr.flatten(), ys=age.flatten(), zs=y1.flatten())

ax.set_xlabel("Overall")

ax.set_ylabel("Age")

ax.set_zlabel("Potential")

#Prediction of potential only from overall,value and age of the players

#Dividing the features

X2=data[["Overall","Age","Value"]].values

y2=data[["Potential"]].values



#splitting data into train test

from sklearn.model_selection import train_test_split



X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.1 )



print(X2_train.shape, y2_train.shape)



from sklearn.linear_model import LinearRegression



regr2=LinearRegression()

regr2.fit(X=X2_train, y=y2_train )

y2_pred=regr2.predict(X2_train)



#intercept,score,RMSE

regr2.intercept_, regr2.score(X2_train, y2_train),np.sqrt(np.mean((y2_train-y2_pred)**2))
#testing the model

y2=regr2.predict(X2_test)



print(regr2.score(X2_test,y2_test))

print("MSE:{}".format(np.sqrt(np.mean((y2_test-y2)**2))))
#Prediction of potential only from overall,wage and age of the players

#Dividing the features

X3=data[["Overall","Age","Wage"]].values

y3=data[["Potential"]].values



#splitting data into train test

from sklearn.model_selection import train_test_split



X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.1 )



print(X3_train.shape, y3_train.shape)



from sklearn.linear_model import LinearRegression



regr3=LinearRegression()

regr3.fit(X=X3_train, y=y3_train )

y3_pred=regr2.predict(X3_train)



#intercept,score,RMSE

regr3.intercept_, regr3.score(X3_train, y3_train),np.sqrt(np.mean((y3_train-y3_pred)**2))
#testing the model

y3=regr2.predict(X3_test)



print(regr3.score(X3_test,y3_test))

print("MSE:{}".format(np.sqrt(np.mean((y3_test-y3)**2))))