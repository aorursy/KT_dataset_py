import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
#Importing The Dataset

sample_submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

print('Train Data: \n\n',train.head(10))

print('*******************************************************')

print('Train Data Shape: ',train.shape)

print('*******************************************************')

print(train.columns)

print('*******************************************************')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

print('Test Data: \n\n',test.head(10))

print('*******************************************************')

print('Test Data Shape: ',test.shape)

print('*******************************************************')

print(test.columns)

print('*******************************************************')
train['train']  = 1

test['train']  = 0

all_dataset = pd.concat([train, test], axis=0,sort=False)
print(all_dataset.head(10))

print(all_dataset.shape)
all_dataset.dtypes.value_counts()
all_dataset.corr()
import seaborn as sns

#correlation matrix

Draw_Corr = all_dataset.corr()

plt.figure(figsize=(30,15))

sns.heatmap(Draw_Corr,annot=True);
plt.figure(figsize=(30,15))

mask = np.zeros(Draw_Corr.shape, dtype=bool)

mask[np.triu_indices(len(mask))] = True

sns.heatmap(Draw_Corr, vmin = -1, vmax = 1, center = 0, annot = True, mask = mask)
#Correlation with output variable

corr_target = abs(Draw_Corr['SalePrice'])

#Selecting Zero correlated features

relevant_features = corr_target[corr_target<0.1]

relevant_features
all_dataset.drop(['Id','MSSubClass','OverallCond','BsmtFinSF2','LowQualFinSF','BsmtHalfBath','3SsnPorch','PoolArea','MiscVal','MoSold','YrSold'], axis=1,inplace=True)

print('All Dataset Shape:',all_dataset.shape)
#Correlation with output variable

corr_target = abs(Draw_Corr['SalePrice'])

#Selecting high correlated features

relevant_features = corr_target[corr_target>=0.5]

relevant_features
plt.figure(figsize=(20,10))

plt.scatter(all_dataset['GrLivArea'],all_dataset['SalePrice'])

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')
plt.figure(figsize=(20,10))

plt.scatter(all_dataset['TotalBsmtSF'],all_dataset['SalePrice'])

plt.xlabel('TotalBsmtSF')

plt.ylabel('SalePrice')
plt.figure(figsize=(20,10))

plt.scatter(all_dataset['YearBuilt'],all_dataset['SalePrice'])

plt.xlabel('YearBuilt')

plt.ylabel('SalePrice')

#scatterplot

sns.set()

cols = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath','TotRmsAbvGrd','GarageCars','GarageArea','SalePrice']

sns.pairplot(all_dataset[cols], size = 2.5)

plt.show();
#missing data

total = all_dataset.isnull().sum().sort_values(ascending=False)

percent = (all_dataset.isnull().sum()/all_dataset.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
import missingno as msno

msno.bar(all_dataset)
all_dataset.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1,inplace=True)
print(all_dataset)

print('All Dataset Shape \n',all_dataset.shape)
all_dataset=pd.get_dummies(all_dataset)
print('All Dataset Shape',all_dataset.shape)
train = all_dataset[all_dataset['train'] == 1]

train = train.drop(['train',],axis=1)



test = all_dataset[all_dataset['train'] == 0]

test = test.drop(['SalePrice'],axis=1)

test = test.drop(['train',],axis=1)
print(train.shape)

print(test.shape)
target=train['SalePrice']

train.drop(['SalePrice'],axis=1,inplace=True)
print(train.shape)

print(test.shape)
X=train.iloc[:,:].values

y=target.values

y=y.reshape(len(y), 1) 

X_test=test.iloc[:,:].values

print('X: ',X[:10,:])

print('X Shape: ',X.shape)

print('Y: ',y)

print('Y_ Shape: ',y.shape)

print('X_test: ',X_test)

print('X_test: ',X_test.shape)
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')

X=imputer.fit_transform(X)

X_test=imputer.fit_transform(X_test)

print(X[:10,:])

print('***********************************')

print(X_test[:10,:])
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score

RidgeRegressionModel = Ridge(alpha=1)

accuracies = cross_val_score(estimator = RidgeRegressionModel, X = X, y = y, cv = 10)



print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
RidgeRegressionModel.fit(X, y)

print('Ridge Regression Train Score is : ' , RidgeRegressionModel.score(X, y))
y_pred_rid = RidgeRegressionModel.predict(X)
from sklearn.metrics import mean_absolute_error

absolute_rid=mean_absolute_error(y,y_pred_rid)

print(absolute_rid)
from sklearn.metrics import mean_squared_error

squared_rid=mean_squared_error(y,y_pred_rid)

print(squared_rid)
from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score

SVR_Linear=SVR(C=1,kernel = 'linear')

accuracies = cross_val_score(estimator = SVR_Linear, X = X, y = y, cv = 10)



print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

print('/////////////////////')
SVR_Linear.fit(X,y)

print('Support Vector Machine Linear Train Score :',SVR_Linear.score(X,y))
y_predict_SVR_Linear=SVR_Linear.predict(X)

print('Y Predict: ',y_predict_SVR_Linear)
from sklearn.metrics import mean_absolute_error

absolute_l=mean_absolute_error(y,y_predict_SVR_Linear)

print(absolute_l)
from sklearn.metrics import mean_squared_error

squared_l=mean_squared_error(y,y_predict_SVR_Linear)

print(squared_l)
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score

dt=DecisionTreeRegressor(max_depth=20,max_features=250,min_samples_split=30)

accuracies = cross_val_score(estimator = dt, X = X, y = y, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

print('/////////////////////')
dt.fit(X,y)

print('DecisionTreeRegressor Train Score :',dt.score(X,y))
y_pred_dt=dt.predict(X)
from sklearn.metrics import mean_absolute_error

absolute_dt=mean_absolute_error(y,y_pred_dt)

print(absolute_dt)
from sklearn.metrics import mean_squared_error

squared_dt=mean_squared_error(y,y_pred_dt)

print(squared_dt)
from sklearn.ensemble import RandomForestRegressor 

from sklearn.model_selection import cross_val_score

rf=RandomForestRegressor(max_depth=13,n_estimators=30,min_samples_split=6)

accuracies = cross_val_score(estimator = rf, X = X, y = y, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

print('/////////////////////')
rf.fit(X,y)

print(' RandomForest Train Score :',rf.score(X,y))

print('RandomForest Classifier Model feature importances are :\n ' , rf.feature_importances_)
y_predict_rf=rf.predict(X)

print('Y Predict: ',y_predict_rf)
from sklearn.metrics import mean_absolute_error

absolute_rf=mean_absolute_error(y,y_predict_rf)

print(absolute_rf)
from sklearn.metrics import mean_squared_error

squared_rf=mean_squared_error(y,y_predict_rf)

print(squared_rf)


models = pd.DataFrame({

    'Mean Squared Error': ['Linear Regression [Ridge]','Support Vector Machines [Regression]','Decision Tree', 

              'Random Forest'],

    'Score': [squared_rid,squared_l,squared_dt,squared_rf]})

models.sort_values(by='Score',ascending=True)
models = pd.DataFrame({

    'Mean Absolute Error': ['Linear Regression [Ridge]','Support Vector Machines [Regression]','Decision Tree', 

              'Random Forest'],

    'Score': [absolute_rid,absolute_l,absolute_dt,absolute_rf]})

models.sort_values(by='Score',ascending=True)
#set ids as PassengerId and predict survival 

ids = sample_submission['Id']

predictions =rf.predict(X_test)



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'Id' : ids, 'SalePrice': predictions })

output.to_csv('submission.csv', index=False)

print(output)