import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv("../input/50_Startups.csv")

X = df.iloc[:,:-1]

y = df.iloc[:,4]
df.head()
#check if there is any null values or not using below command. If yes, then use Imputer class to handle missing values.

df.isnull().sum()
#check for any categorical variable (if any)

#Encoding categorical data in this case (independent variable)

from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()

X.iloc[:,3] = labelencoder_X.fit_transform(X.iloc[:,3])
#encoding the independent variables

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder



ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')

X = ct.fit_transform(X)

X = np.array(X)

X
#Alert! Avoid the dummy variable trap by removing any one dummy variable

X = X[:, 1:]
from sklearn.model_selection import train_test_split  #(for python2)

#from sklearn.model_selection import train_test_split  (for python3)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
#no feature scaling is reuqired as Multiple Linear Refression algorithm take care by itself
#fit data into the model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
y_compare = pd.DataFrame(

    {'Original Profit': y_test,

     'Predicted Profit': y_pred,

     'Residual Error' : y_test-y_pred

    }).reset_index()

y_compare