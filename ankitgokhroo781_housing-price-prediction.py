# This Python 3 environment comes with many helpful analytics libraries installed  

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python  

# For example, here's several helpful packages to load in  



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)  

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/kc_house_data.csv',)



data.head()
data.info()
data.columns
data.describe()
data.isnull().any(axis=0)
data=data.drop(['id','date','sqft_lot','zipcode','sqft_living15','sqft_lot15','lat','long'],axis=1)
data.head()
data['waterfront'].value_counts()
data['view'].value_counts()
data['grade'].value_counts()
data['condition'].value_counts()
#dataset= dataset.get_dummies(dataset)

features = data.iloc[:, 1:].values

labels = data.iloc[:, 0].values
features[0]
labels[0]
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(features_train, labels_train)
print(regressor.intercept_)  

print (regressor.coef_)
# Predicting the Test set results

Pred = regressor.predict(features_test)




print (pd.DataFrame(Pred, labels_test))

# Getting Score for the Multi Linear Reg model

Score_train = regressor.score(features_train, labels_train)

Score_train
Score_test = regressor.score(features_test, labels_test)

Score_test


x =[3,2,1145,1,0,1,1,3,1146,1367,1900,1]

x = np.array(x)

x = x.reshape(1,12)

regressor.predict(x)
data.head(1)
from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(labels_test, Pred))  

print('Mean Squared Error:', metrics.mean_squared_error(labels_test, Pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(labels_test, Pred)))  



print (np.mean(labels))
from sklearn.tree import DecisionTreeRegressor  

regressor_dt = DecisionTreeRegressor()  

regressor_dt.fit(features_train, labels_train)  



labels_pred = regressor_dt.predict(features_test)

df=pd.DataFrame({'Actual':labels_test, 'Predicted':labels_pred})  
df.head()
# Getting Score for the Multi Linear Reg model

Score_train_dt = regressor_dt.score(features_train, labels_train)

Score_train_dt
Score_test_dt = regressor_dt.score(features_test, labels_test)

Score_test_dt
#train the model

from sklearn.ensemble import RandomForestRegressor



regressor_rf = RandomForestRegressor(n_estimators=25, random_state=0)  

regressor_rf.fit(features_train, labels_train)  

labels_pred = regressor_rf.predict(features_test)  
Score_test_rf = regressor_rf.score(features_test, labels_test)

Score_test_rf
Score_train_rf = regressor_rf.score(features_train, labels_train)

Score_train_rf
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()  

features_train = sc.fit_transform(features_train)  

features_test = sc.transform(features_test)  
#train the model

from sklearn.ensemble import RandomForestRegressor



regressor_rf = RandomForestRegressor(n_estimators=25, random_state=0)  

regressor_rf.fit(features_train, labels_train)  

labels_pred = regressor_rf.predict(features_test)  
Score_test_rf = regressor_rf.score(features_test, labels_test)

Score_test_rf

Score_train_rf = regressor_rf.score(features_train, labels_train)

Score_train_rf
#Lets build the models now

#Multiple Linear Regression

from sklearn.model_selection import cross_val_score





MSEs = cross_val_score(regressor_rf, features_train, labels_train, scoring='neg_mean_squared_error', cv=5)



mean_MSE = np.mean(MSEs)



print(mean_MSE)





#Ridge Regression

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge



#alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]



ridge = Ridge(alpha=20)



#parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}



ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)



ridge_regressor.fit(features_train, labels_train)





ridge_regressor.best_params_


ridge_regressor.best_score_

Score_test_rf = ridge_regressor.score(features_test, labels_test)

Score_test_rf
rr = Ridge(alpha=0.01) # higher the alpha value, more restriction on the coefficients; low alpha > more generalization, coefficients are barely

# restricted and in this case linear and ridge regression resembles

rr.fit(X_train, y_train)

rr100 = Ridge(alpha=100) #  comparison with alpha value

rr100.fit(X_train, y_train)

train_score=lr.score(X_train, y_train)

test_score=lr.score(X_test, y_test)

Ridge_train_score = rr.score(X_train,y_train)

Ridge_test_score = rr.score(X_test, y_test)

Ridge_train_score100 = rr100.score(X_train,y_train)

Ridge_test_score100 = rr100.score(X_test, y_test)
# Import Ridge regression from sklearn

from sklearn.linear_model import Ridge

# Evaluate model performance using root mean square error

from sklearn.metrics import mean_squared_error

rmse=[]

# check the below alpha values for Ridge Regression

alpha=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]



for alph in alpha:

    ridge=Ridge(alpha=alph, copy_X=True, fit_intercept=True)

    ridge.fit(features_train, labels_train)

    predict=ridge.predict(features_test)

    rmse.append(np.sqrt(mean_squared_error(predict, labels_test)))

print(rmse)

plt.scatter(alpha, rmse)
#import dataset and make feature and label

x=my_data.iloc[:,1:].values

y=my_data.iloc[:,0:1].values



#split data into train and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)



#fitting simple linear regression to the training set

from sklearn.linear_model import LinearRegression

reg=LinearRegression()

reg.fit(X_train,y_train)



#visualising the training set results

plt.scatter(X_train,y_train,color='Red',marker='+')

plt.plot(X_train,reg.predict(X_train),color='blue')

plt.xlabel('area',fontsize=20)

plt.ylabel('price',fontsize=20)

plt.title('area vs price')

plt.show()

##visualising the testing set results

plt.scatter(X_test,y_test,color='Red',marker='+')

plt.plot(X_train,reg.predict(X_train),color='blue')

plt.xlabel('area',fontsize=20)

plt.ylabel('price',fontsize=20)

plt.title('area vs price')

plt.show()

# regression coefficients 

print('Coefficients: \n', reg.coef_)
# variance score: 1 means perfect prediction 

print('Variance score: {}'.format(reg.score(X_test, y_test))) 

  
score=reg.score(X_test,y_test)

score
#predicting the test set result

y_pred=reg.predict(X_test)

y_pred
# plot for residual error 

  

## setting plot style 

plt.style.use('fivethirtyeight') 



## plotting residual errors in training data 

plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, color = "blue", s = 10, label = 'Train data')



## plotting residual errors in test data 

plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, color = "red", s = 10, label = 'Test data') 

  
## plotting line for zero residual error 

plt.hlines(y=0, xmin = 0, xmax = 900, linewidth = 50) 

  

## plotting legend 

plt.legend(loc = 'upper right') 

  

## plot title 

plt.title("Residual errors") 

  

## function to show plot 

plt.show() 
