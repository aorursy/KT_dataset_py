# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline
housing=pd.read_csv('../input/housing.csv')
housing.head()
housing.hist(bins=50, figsize=(20,15))
plt.show()
print("Description : ",housing.describe())
X=housing.iloc[:,0:-1].values
y=housing.iloc[:,housing.shape[1]-1].values
housing.info()
from sklearn.preprocessing import Imputer
missingValueImputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
X[:,2:-1]=missingValueImputer.fit_transform(X[:,2:-1])
pp=pd.DataFrame(X)
pp.info()
housing.plot(kind="scatter", x="longitude", y="latitude", title="Cal districts", alpha=0.4, 
             s=housing["population"]/100, label="Population",
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar="True")
plt.legend()
from sklearn.preprocessing import LabelEncoder
X_labelencoder= LabelEncoder()
X[:,8]=X_labelencoder.fit_transform(X[:,8])
X[:,8]
from sklearn.preprocessing import OneHotEncoder
#Instaciate a new encoder
X_onehotencoder=OneHotEncoder(categorical_features=[8])
X=X_onehotencoder.fit_transform(X).toarray()
X

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
independent_scalar=StandardScaler()
X_train=independent_scalar.fit_transform(X_train)
X_test=independent_scalar.transform(X_test)
from sklearn.linear_model import LinearRegression
regressoragent = LinearRegression()
regressoragent.fit (X_train, y_train)
predictValues = regressoragent.predict(X_test)
predictValues
#Accuracy of the model for training data 
regressoragent.score(X_train,y_train)
# Accuracy of the model for testing data
regressoragent.score(X_test,y_test)
y_pred = regressoragent.predict(X_test)
from sklearn.metrics import mean_squared_error
from math import sqrt
meanSquaredError=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:{}".format(meanSquaredError))
c=sqrt(meanSquaredError)
print("RMSE={}".format(c))
from sklearn.tree import DecisionTreeRegressor
DTregressor = DecisionTreeRegressor (random_state = 0)
DTregressor.fit (X_train, y_train)
y_predict_train_DTregressor=DTregressor.predict(X_train)
y_predict_test_DTregressor=DTregressor.predict(X_test)
y_predict_test_DTregressor
from sklearn.metrics import mean_squared_error
from math import sqrt
modelPred = DTregressor.predict(X_test)
print(modelPred)
print("Number of predictions:",len(modelPred))

RmeanSquaredError=np.sqrt(mean_squared_error(y_test,y_predict_test_DTregressor))
print("Root Mean Squared Error:{}".format(RmeanSquaredError))

from sklearn.ensemble import RandomForestRegressor
RFreg=RandomForestRegressor(random_state=0)
RFreg.fit(X_train,y_train)
y_predict_train_RFreg=RFreg.predict(X_train)
y_predict_test_RFreg=RFreg.predict(X_test)
from sklearn.metrics import mean_squared_error
from math import sqrt
modelPred = RFreg.predict(X_test)
print(modelPred)
print("Number of predictions:",len(modelPred))
meanSquaredError=np.sqrt(mean_squared_error(y_test,y_predict_test_RFreg))
print("Root Mean Squared Error:{}".format(meanSquaredError))

dict={'Linear Regression':c,'Decision Tree Regression':RmeanSquaredError,'Random Forest Regression':meanSquaredError}
results=pd.DataFrame(dict,index=['RMSE'])
results
results.plot(kind='barh',figsize=(10,10))
plt.title('Comparing the 3 test cases')
plt.show()

