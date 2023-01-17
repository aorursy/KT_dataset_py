# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.shape
train.head()
_temp=pd.DataFrame(train.isnull().sum()).reset_index()

non_null_columns=_temp[_temp[0]==0]['index'].tolist()
train_not_null=train[non_null_columns]
train_not_null.shape
train_not_null.head()
train_not_null['MSZoning'].value_counts()
import seaborn as sns

sns.scatterplot(x='MSZoning',y='SalePrice',data=train_not_null)
test['MSZoning'].value_counts()
#treating MSZoning
test['MSZoning'].isnull().sum()
test['MSZoning'].fillna('RL',inplace=True)
#Treating MSSubClass

train_not_null['MSSubClass'].value_counts()
test['MSSubClass'].value_counts()
def change_mssubclass(value):

    if (value==20) or (value==30) or (value==40) or (value==120):

        return 20

    elif (value==45) or (value==50) or (value==150):

        return 45

    elif (value==60) or (value==70) or (value==160) or (value==75) or (value==190):

        return 60

    elif (value==80) or (value==85) or (value==180):

        return 80

    else:

        return value
test['MSSubClass']=test['MSSubClass'].apply(lambda x: change_mssubclass(x))

train_not_null['MSSubClass']=train_not_null['MSSubClass'].apply(lambda x: change_mssubclass(x))
train_not_null['MSSubClass'].value_counts()
sns.scatterplot(x='MSSubClass',y='SalePrice',data=train_not_null)
#Treating LotArea
train_not_null['LotArea']=np.log(train_not_null['LotArea'])

test['LotArea']=np.log(test['LotArea'])
sns.scatterplot(x='LotArea',y='SalePrice',data=train_not_null)
sns.distplot(test['LotArea'])
sns.scatterplot(x='OverallQual',y='SalePrice',data=train_not_null)
train_not_null['OverallQual'].value_counts()
test['OverallQual'].value_counts()
sns.scatterplot(x='YearBuilt',y='SalePrice',data=train_not_null)
#Treating ground living area
train_not_null['GrLivArea']=np.log(train_not_null['GrLivArea'])

test['GrLivArea']=np.log(test['GrLivArea'])
sns.scatterplot(x='GrLivArea',y='SalePrice',data=train_not_null)
sns.distplot(test['GrLivArea'])
#Treating TotalBsmtSF
sns.distplot(train_not_null['TotalBsmtSF']**(1/2))
sns.distplot(test['TotalBsmtSF'].fillna(0)**(1/2))
train_not_null['TotalBsmtSF']=train_not_null['TotalBsmtSF']**(1/2)

test['TotalBsmtSF']=(test['TotalBsmtSF'].fillna(0))**(1/2)
sns.scatterplot(x='TotalBsmtSF',y='SalePrice',data=train_not_null)
sns.scatterplot(x='Neighborhood',y='SalePrice',data=train_not_null)
#Treating 1stflrsf
train_not_null['1stFlrSF']=train_not_null['1stFlrSF']**(1/2)

test['1stFlrSF']=(test['1stFlrSF'].fillna(0))**(1/2)
sns.scatterplot(x='1stFlrSF',y='SalePrice',data=train_not_null)
sns.scatterplot(x='GarageArea',y='SalePrice',data=train_not_null)
sns.scatterplot(x='TotRmsAbvGrd',y='SalePrice',data=train_not_null)
train_not_null['Fence']=train['Fence'].fillna('nofence')
train['FullBath'].value_counts()
train_not_null['LotFrontage']=(train['LotFrontage'].interpolate())**(1/4)

test['LotFrontage']=(test['LotFrontage'].interpolate())**(1/4)
sns.distplot((train_not_null['EnclosedPorch'])**(1/1.9))
sns.scatterplot(x='Fireplaces',y='SalePrice',data=train_not_null)
sns.distplot(train_not_null['LotFrontage'])
sns.distplot((test['LotFrontage']))
final_df=train_not_null[['TotalBsmtSF','GrLivArea','YearBuilt','OverallQual','LotArea','SalePrice','1stFlrSF','GarageArea','LotFrontage','TotRmsAbvGrd','Neighborhood','MSSubClass','MSZoning','HouseStyle','FullBath']]
final_df.dtypes
final_df['OverallQual']=final_df['OverallQual'].astype('object')

final_df['MSSubClass']=final_df['MSSubClass'].astype('object')

final_df['TotRmsAbvGrd']=final_df['TotRmsAbvGrd'].astype('object')

final_df['FullBath']=final_df['FullBath'].astype('object')

#final_df['Fireplaces']=final_df['Fireplaces'].astype('object')

final_df.dtypes
final_df=pd.get_dummies(final_df,drop_first=True)
final_df.head()
sns.heatmap(final_df.corr(),linewidths=.15)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns #for plotting

from sklearn.ensemble import RandomForestClassifier #for the model

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz #plot tree

from sklearn.metrics import roc_curve, auc #for model evaluation

from sklearn.metrics import classification_report #for model evaluation

from sklearn.metrics import confusion_matrix #for model evaluation

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split #for data splitting

import eli5 #for purmutation importance

from eli5.sklearn import PermutationImportance

from sklearn.preprocessing import RobustScaler

import shap #for SHAP values

from pdpbox import pdp, info_plots #for partial plots

np.random.seed(123) #ensure reproducibility

#split the data

X_train, X_test, y_train, y_test = train_test_split(final_df.drop(['SalePrice'], 1), final_df['SalePrice'], test_size = .2, random_state=10) 
scaler=RobustScaler()

X_train_scaled=scaler.fit_transform(X_train)

X_test_scaled=scaler.transform(X_test)

X_train_scaled.shape
from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler

from math import sqrt

from sklearn.metrics import make_scorer
def rmse(y, y_pred):

    return np.sqrt(np.mean(np.square(y - y_pred)))
from xgboost import XGBRegressor

gsc = GridSearchCV(

        estimator=XGBRegressor(),

        param_grid={

            'max_depth': range(2,7),

            'n_estimators': (100,150,200,250,300,500, 1000,1500,2000,2500,3000),

        },

        cv=5, scoring='neg_mean_squared_log_error', verbose=0,n_jobs=-1)

    

grid_result = gsc.fit(X_train_scaled, y_train)

best_params = grid_result.best_params_
best_params


xgboost = XGBRegressor(learning_rate=0.01,n_estimators=best_params["n_estimators"],

                                     max_depth=best_params["max_depth"], min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)

xgboost.fit(X_train_scaled,y_train)
rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],random_state=False, verbose=False)

rfr.fit(X_train_scaled, y_train)
# Perform K-Fold CV

scores = cross_val_score(rfr, X_train_scaled, y_train, cv=10, scoring='neg_mean_squared_log_error')

print(scores.mean())
scores.std()
# Perform K-Fold CV

scores = cross_val_score(xgboost, X_train_scaled, y_train, cv=10, scoring='neg_mean_squared_log_error')

print(scores.mean())
scores.std()
from sklearn.model_selection import learning_curve

# Create CV training and test scores for various training set sizes

train_sizes, train_scores, test_scores = learning_curve(xgboost, 

                                                        X_train_scaled, 

                                                        y_train,

                                                        # Number of folds in cross-validation

                                                        cv=10,

                                                        # Evaluation metric

                                                        scoring='neg_mean_squared_log_error',

                                                        # Use all computer cores

                                                        n_jobs=-1, 

                                                        # 50 different sizes of the training set

                                                        train_sizes=[100,200,300 ,400,500,600,700,900,1000,1051])
train_scores_mean = -train_scores.mean(axis = 1)

test_scores_mean = -test_scores.mean(axis =1)
train_scores_mean.mean()
test_scores_mean.mean()
plt.style.use('seaborn')

plt.plot(train_sizes, train_scores_mean, label = 'Training error')

plt.plot(train_sizes, test_scores_mean, label = 'Validation error')

plt.ylabel('MSE', fontsize = 14)

plt.xlabel('Training set size', fontsize = 14)

plt.title('Learning curves for a linear regression model', fontsize = 18, y = 1.03)

plt.legend()

plt.ylim(0,.1)
y_predict = xgboost.predict(X_test_scaled)
rmse(y_test,y_predict)
from sklearn.metrics import mean_squared_log_error

mean_squared_log_error(y_test, y_predict)
final_test=test[['TotalBsmtSF','GrLivArea','YearBuilt','OverallQual','LotArea','1stFlrSF','GarageArea','LotFrontage','TotRmsAbvGrd','Neighborhood','MSSubClass','MSZoning','HouseStyle','FullBath']]
final_test['OverallQual']=final_test['OverallQual'].astype('object')

final_test['MSSubClass']=final_test['MSSubClass'].astype('object')

final_test['TotRmsAbvGrd']=final_test['TotRmsAbvGrd'].astype('object')

final_test['FullBath']=final_test['FullBath'].astype('object')

#final_df['Fireplaces']=final_df['Fireplaces'].astype('object')

final_test_df=pd.get_dummies(final_test,drop_first=True)
final_test_df.head()
final_test_df_scaled=scaler.transform(final_test_df)
final_predictions = xgboost.predict(final_test_df_scaled)
submission=pd.DataFrame(test['Id'])

submission['SalePrice']=final_predictions
submission.to_csv('xgboost3.csv',index=False)