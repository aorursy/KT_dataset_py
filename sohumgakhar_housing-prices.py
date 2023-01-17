# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print(train.shape)

print(test.shape)
# Concatenate both the training and the testing data together to make it easy to perform operatiions on both

data = pd.concat([train,test],axis = 0)

print(data.shape)
data.info()
# finding percentage of missing values in each feature that exceeds 20%

print((data.isnull().sum()/len(data)*100).round(2)[(data.isnull().sum()/len(data)*100) >=20])
# Features with missing values more than 30% can be dropped and will hardly play any significance in the modelling



features_to_drop = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature','Id']

data = data.drop(features_to_drop,axis = 1)
data.shape

# We have dropped 6 collumns
# Create new features based on intuition

data['TotalArea'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF'] + data['GrLivArea'] +data['GarageArea']



data['Bathrooms'] = data['FullBath'] + data['HalfBath']*0.5 



data['Year average']= (data['YearRemodAdd'] + data['YearBuilt'])/2
# Collumns having at least one Nan value

print("Columns having at least one NaN Value:" + str((data.isnull().sum()/len(data)*100).round(2)[(data.isnull().sum()/len(data)*100) >0.0].count()))
print("Columns having at least one NaN Value:" + str((data.isnull().sum()/len(data)*100).round(2)[(data.isnull().sum()/len(data)*100) >0.0].sort_values(ascending = False)))
# Deal with the NaN values now!

nan_val = (data.isnull().sum()/len(data)*100).round(2)[(data.isnull().sum()/len(data)*100) >0.0].sort_values(ascending = False).index.to_list()

nan_val
# Finding Data types of the features

data_nan_val = (data[nan_val].dtypes ==object)

data_nan_val = data_nan_val.astype('str')

data_nan_val
obj_nan_val = data_nan_val[data_nan_val == 'True'].index.to_list()

num_nan_val = data_nan_val[data_nan_val == 'False'].index.to_list()

print("Object data type Features:\n")

print(obj_nan_val)

print("\n")

print("Numeric data type Features:\n")

print(num_nan_val)
num_nan_val.remove('SalePrice')
# USing LAbel Encoder we will not transform the object type inputs into numerical form

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



for i in range(data.shape[1]):

    if data.iloc[:,i].dtypes == object:

        le.fit(list(data.iloc[:,i].values))

        data.iloc[:,i] = le.transform(list(data.iloc[:,i].values))

# Confirm that the code is working properly

print(data['GarageCond'].unique())
# Fill the NaN values with the mode and the median for the object_NaN and the numeric_NaN values respectively

data[num_nan_val] = data[num_nan_val].fillna(data[num_nan_val].median())



for j in data[obj_nan_val]:

    data[j] = data[j].fillna(data[j].mode())  
# Split the data back into test data and train data

test_data = data[data.SalePrice.isnull() == True]

test_data = test_data.drop("SalePrice",axis = 1)

train_data = data.dropna(axis = 0)
print("Shape of training data after data cleaning:")

print(train_data.shape)

print("Shape of testing data after data cleaning:")

print(test_data.shape)

# We have dropped out 6 columns
print(train_data.SalePrice.describe().round(2))

plt.figure(figsize = (7,7))

sns.distplot(train_data.SalePrice,color = 'red')
# Log transform the target variable for handling the outliers

train_data["SalePrice"] = np.log(train_data['SalePrice'])

plt.figure(figsize = (7,7))

sns.distplot(train_data.SalePrice,color = 'yellow')
train_data.describe()
train_data.hist(figsize = (35,30),color = 'red');
data_corr = train_data.corr()

data_corr = data_corr.SalePrice

data_corr = data_corr.drop('SalePrice')
strong_corr_feat = data_corr[abs(data_corr) > 0.3].round(2).sort_values(ascending = False)  

# abs() to only take the magnitude

print(strong_corr_feat)

len(strong_corr_feat)

# 29 features that have an impact on the selling prices
Y_train = train_data['SalePrice']

X_train = train_data.drop('SalePrice',axis = 1)

X_test = test_data
# Feature Importance using RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 80)

rf.fit(X_train,Y_train)

ranking = np.argsort(-rf.feature_importances_)

# np.argsort returns indices of features that are of maximum importance

f, ax = plt.subplots(figsize=(18, 12))

sns.barplot(x=rf.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')

ax.set_xlabel("Feature Importance")

plt.tight_layout()

plt.show()
X_train = X_train.iloc[:,ranking[:29]]

X_test = X_test.iloc[:,ranking[:29]]
# When dealing with pandas dataframes, You cannot directly slice up by using indices. 'df.iloc' allows slicing by using indices
# Now we are going to find out how each and every feature influences the SalePrice in it's own unique way!

fig = plt.figure(figsize = (12,9))

for i in np.arange(29):

    ax = fig.add_subplot(5,6,i + 1)

    sns.regplot(x = X_train.iloc[:,i],y=Y_train)

plt.tight_layout()

plt.show()
sns.regplot(x = X_train.iloc[:,0],y=Y_train)

plt.show()
# Remove the outliers

X = X_train

X['SalePrice'] = Y_train

X = X.drop(X[(X['YearBuilt']<1900) & (10.75>X['SalePrice']) & (X['SalePrice']>13.0)].index)

X = X.drop(X[(X['TotalArea']>10000)&(10.5<X['SalePrice'])].index)

X = X.drop(X[(X['GrLivArea']>3500) & (10.5>X['SalePrice'])].index)

X = X.drop(X[(X['TotalBsmtSF']>2900) & (10.5>X['SalePrice'])].index)

X = X.drop(X[(200>X['GarageArea']) & (X['GarageArea']>1100) & (X['SalePrice']<10.7)].index)

Y_train = X['SalePrice']

X_train = X.drop(['SalePrice'],axis = 1)
import xgboost as xgb

from sklearn.model_selection import GridSearchCV



print("Parameter optimization")

xgb_model = xgb.XGBRegressor()

reg_xgb = GridSearchCV(xgb_model,

                   {'max_depth': [2,4,6],

                    'n_estimators': [50,100,200]}, verbose=1)

reg_xgb.fit(X_train, Y_train)

print(reg_xgb.best_score_)

print(reg_xgb.best_params_)
from sklearn.linear_model import Lasso

import sklearn.model_selection as ms

parameters= {'alpha':[0.0001,0.0009,0.001,0.01,0.1,1,10],

            'max_iter':[100,500,1000]}





lasso = Lasso()

lasso_model = ms.GridSearchCV(lasso, param_grid=parameters, scoring='neg_mean_squared_error', cv=10)

lasso_model.fit(X_train,Y_train)



print('The best value of Alpha is: ',lasso_model.best_params_)

print(lasso_model.best_score_)
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor



def create_model(optimizer='adam'):

    model = Sequential()

    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))

    model.add(Dense(16, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))



    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model



model = KerasRegressor(build_fn=create_model, verbose=0)

# define the grid search parameters

optimizer = ['SGD','Adam']

batch_size = [10, 30, 50]

epochs = [10, 50, 100]

param_grid = dict(optimizer=optimizer, batch_size=batch_size, epochs=epochs)

reg_dl = GridSearchCV(estimator=model, param_grid=param_grid)

reg_dl.fit(X_train, Y_train)



print(reg_dl.best_score_)

print(reg_dl.best_params_)
model = Sequential()

model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))

model.add(Dense(16, kernel_initializer='normal', activation='relu'))

model.add(Dense(1, kernel_initializer='normal'))

model.compile(optimizer='Adam',loss='mean_squared_error')

model.fit(X_train,Y_train,batch_size = 10,epochs=100)
pred2 = np.exp(model.predict(X_test))

pred2 = pred2.reshape(-1,)

pred2.shape
# SVR

from sklearn.svm import SVR



reg_svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,

                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],

                               "gamma": np.logspace(-2, 2, 5)})

reg_svr.fit(X_train, Y_train)



print(reg_svr.best_score_)

print(reg_svr.best_params_)
# second feature matrix

X_train2 = pd.DataFrame( {'XGB': reg_xgb.predict(X_train),

     'NN': reg_dl.predict(X_train).ravel(),

     'SVR': reg_svr.predict(X_train),

    })

X_train2.head()
# prediction using the test set

from sklearn import linear_model



reg = linear_model.LinearRegression()

reg.fit(X_train2, Y_train)



X_test2 = pd.DataFrame( {'XGB': reg_xgb.predict(X_test),

     'DL': reg_dl.predict(X_test).ravel(),

     'SVR': reg_svr.predict(X_test),

    })



# Don't forget to convert the prediction back to non-log scale

y_pred = np.exp(reg.predict(X_test2))

lasso = Lasso(alpha = 0.0001,max_iter = 1000)

lasso.fit(X_train,Y_train)

pred1= np.exp(lasso.predict(X_test))
xgb1 = xgb.XGBRegressor(max_depth = 2,n_estimators=200)

xgb1.fit(X_train,Y_train)

y_pred = np.exp(xgb1.predict(X_test))
y_pred.shape

test_Id = test['Id']

submission = pd.DataFrame({'Id':test_Id,"SalePrice":pred2})

submission.to_csv('houseprice1.csv',index = False)