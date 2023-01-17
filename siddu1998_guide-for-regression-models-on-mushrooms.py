import numpy as np

import pandas as pd



#Importing the dataset

dataset=pd.read_csv("../input/mushrooms.csv") 

#the variable dataset will now contain records os mushroom.cvs
#Encoding the Categorical Data

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

for col in dataset.columns:

    dataset[col] = labelencoder.fit_transform(dataset[col])

#Splitting the data into dependet and independent variables

X=dataset.iloc[:,1:].values

y=dataset.iloc[:,0].values
#Splitting the data into training

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#Using Logistic Regression

from sklearn.linear_model import LogisticRegression

regressor=LogisticRegression()

regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

print('Using Simple Logistic Regression we achieve an accuracy of',np.mean(y_test==y_pred)*100)
#Using Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor

regressor=DecisionTreeRegressor(random_state=0)

regressor.fit(X,y)

y_pred=regressor.predict(X_test)

print("Using Decision Tree Regression we achieve an accuracy of",np.mean(y_pred==y_test)*100)





#Using Random Forest Regression

from sklearn.ensemble import RandomForestRegressor

regressor=RandomForestRegressor(n_estimators=300,random_state=0)

regressor.fit(X,y)

y_pred=regressor.predict(X_test)

print('Using Random Forest Regression we achieve an accuray of',np.mean(y_test==y_pred)*100)
#Using Multipe Linear Regression with a combination Back-tracking

import statsmodels.formula.api as sm

X=np.append(arr=np.ones((8124, 1)).astype(int),values=X,axis=1)

X_opt = X[:, :]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit();

regressor_OLS.summary()

pvalues = regressor_OLS.pvalues

sl = 0.05

while 1:

    high = pvalues[0]

    highPos = 0

    for i in range(1, len(pvalues)):

        if pvalues[i] > high:

            high = pvalues[i]

            highPos = i

    # Check if > SL and if so remove the column

    if (high > sl):

        X_opt = np.delete(X_opt, highPos, 1)

        regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit();

        pvalues = regressor_OLS.pvalues

    else:

        regressor_OLS.summary()

        break

print(regressor_OLS.summary())

X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]

X_train ,X_test ,y_train, y_test=train_test_split(X_opt,y,test_size=0.2,random_state=0)

regressor.fit(X_train,y_train)

y_pred_with_multi=regressor.predict(X_test)

print('Using Multiple Linear Regression we achieve an accuracy of',np.mean(y_test==y_pred)*100)


