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
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, confusion_matrix, accuracy_score

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV



from sklearn import tree, ensemble, linear_model



from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor

from xgboost.sklearn import XGBRegressor

from sklearn import svm

from sklearn.neural_network import MLPRegressor
# data

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

sub = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



testing = test.copy()



target = "SalePrice"



print(train.shape, test.shape, sub.shape)

train.info(), test.info()
df_num = train.select_dtypes(include = ['float64', 'int64'])



for i in range(0, len(df_num.columns), 6):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+6],

                y_vars=['SalePrice'], kind='reg')
train.drop(train[train.GrLivArea>4000].index, inplace = True)

train.reset_index(drop = True, inplace = True)



train.drop(train[train.BsmtFinSF1>4000].index, inplace = True)

train.reset_index(drop = True, inplace = True)

      

train.drop(train[train.TotalBsmtSF>4000].index, inplace = True)

train.reset_index(drop = True, inplace = True)

         

train.drop(train[train.LotArea>200000].index, inplace = True)

train.reset_index(drop = True, inplace = True)
df_num = train.select_dtypes(include = ['float64', 'int64'])



for i in range(0, len(df_num.columns), 6):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+6],

                y_vars=['SalePrice'], kind='reg')
# keeping only those object features which are missing not more then 80% of it's data.



def drop_invalid_objects(x):

    count_null = x.select_dtypes(include= 'object').isna().sum()

    count_null = count_null[count_null > 0]



    ratio = x['Id'].count()*0.2



    drop_NaN_object = pd.DataFrame(count_null[count_null > ratio])

    drop_NaN_object = drop_NaN_object.reset_index().filter(['index']).set_index('index').transpose()

    drop_NaN_object = list(drop_NaN_object)



    x.drop(columns = x[drop_NaN_object], inplace = True)



    count_null = x.select_dtypes(include= 'object').isna().sum()

    count_null = count_null[count_null > 0]

    print(count_null)



drop_invalid_objects(train)

print('='*30)

drop_invalid_objects(test)
count_null = train.select_dtypes(include= 'object').isna().sum()

count_null = pd.DataFrame(count_null[count_null > 0])

count_null = count_null.reset_index().filter(['index']).set_index('index').transpose()

count_null = list(count_null)



for column in count_null:

    train[column] = train[column].fillna(train[column].mode()[0])



for column in count_null:

    test[column] = test[column].fillna(test[column].mode()[0])    
train.fillna(train.mean(), inplace=True)

test.fillna(test.mean(), inplace=True)



print(test.shape, train.shape)

train.info(), test.info()

train.tail()
train.describe()
train_dummy = list(train.select_dtypes(include='object').columns)

test_dummy = list(test.select_dtypes(include='object').columns)



train = pd.concat([train, pd.get_dummies(train[train_dummy])], axis = 1, sort = False)

train.drop(columns = train[train_dummy], inplace = True)



test = pd.concat([test, pd.get_dummies(test[test_dummy])], axis = 1, sort = False)

test.drop(columns = test[test_dummy], inplace = True)



train.info()

print('='*50)

test.info()
print(train.shape, test.shape)



train.tail(10)
#plt.figure(figsize=(30,20))

cor = train.corr()

#sns.heatmap(cor, annot=True, cmap=plt.cm.Reds);



#Correlation with output variable

cor_target = abs(cor[target])



#Selecting highly correlated features

correlated_features = cor_target[cor_target >= 0.02]

print('Amount of corr features: ', len(correlated_features))
correlated_features = pd.DataFrame(correlated_features)

correlated_features.reset_index(level=0, inplace=True)



train = train[correlated_features['index']]

train = train.reindex(sorted(train.columns), axis=1)



train.info(), test.info()
corr_test_features = correlated_features[correlated_features != target].dropna(axis = 0).drop(columns = target)

corr_test_features = corr_test_features.set_index('index').transpose()



frames = [test, corr_test_features]



test2 = pd.concat(frames, axis = 0, join = 'inner', sort = True)



test2.info()
feature_differences = train.columns.difference(test2.columns)

feature_differences = pd.DataFrame(feature_differences).set_index(0).transpose().drop(columns = target)



frames = [test2, feature_differences]



full_test = pd.concat(frames, axis = 0, join = 'outer', sort = False).fillna(-99999)

full_test = full_test.reindex(sorted(full_test.columns), axis=1)

full_test = full_test.fillna(-99999)



print(train.columns.difference(test2.columns))

print(train.columns.difference(full_test.columns))



print('\nTEST dataset shape: ', test.shape)

print('TEST2 dataset shape: ', test2.shape)

print('Full_test dataset shape: ', full_test.shape)

print('\nTRAIN dataset shape: ', train.shape)
# Train and Test dataset split

y = train[target]

x = train.drop(columns = target)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 42)



x_train.shape, x_test.shape, full_test.shape
xgb = XGBRegressor()

xgb_params = {

          'n_estimators':[n for n in range(900, 1200, 100)],

          'max_depth': [n for n in range(2, 4)],

          'random_state' : [42]

            }



xgb_model = GridSearchCV(xgb, param_grid = xgb_params, cv = 7, n_jobs = -1)

xgb_model.fit(x_train, y_train)



xgb_predictions = xgb_model.predict(x_test)

print("r2_score: " + '%.3f' % r2_score(y_test, xgb_predictions))



print("Best Hyper Parameters: ",xgb_model.best_params_)

print("Best Score: " + '%.3f' % xgb_model.best_score_)
# Plot feature importance

feature_importance = pd.Series(xgb_model.best_estimator_.feature_importances_, index=x_train.columns)



plt.figure(figsize=(10,8))

feature_importance.nlargest(15).sort_values(ascending = True).plot(kind='barh')

plt.show()
GB = ensemble.GradientBoostingRegressor()

GB_params = {

          'n_estimators':[1000],

          'max_depth': [n for n in range(2, 4)],

          'subsample' : [0.8],

          'random_state' : [42]



            }



GB_model = GridSearchCV(GB, param_grid = GB_params, cv = 5, n_jobs = -1)

GB_model.fit(x_train, y_train)



gbt_predictions = GB_model.predict(x_test)

print("r2_score: " + '%.3f' % r2_score(y_test, gbt_predictions))



print("Best Hyper Parameters: ",GB_model.best_params_)

print("Best Score: " + '%.3f' % GB_model.best_score_)
# Plot feature importance

feature_importance = pd.Series(GB_model.best_estimator_.feature_importances_, index=x_train.columns)



plt.figure(figsize=(10,8))

feature_importance.nlargest(15).sort_values(ascending = True).plot(kind='barh')

plt.show()
def submision_file(model, model_name):

    prediction = model.predict(full_test)

    submision = pd.DataFrame({'Id':testing['Id'],'SalePrice':prediction})

    

    #creating submission file

    file_name = 'House_SalePrices_'+model_name+'.csv'

    submision.to_csv(file_name,index=False)

    print('\nSaved file: ' + file_name)



submision_file(GB_model, 'GBT')

submision_file(xgb_model, 'XGB')