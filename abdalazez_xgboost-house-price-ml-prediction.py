# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',index_col='Id')

data_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',index_col='Id')

data_test.tail()
data_train.describe()
# Get  columns whose data type is object i.e. string

###filteredColumns_Train = data_train.dtypes[data_train.dtypes == np.object]

###filteredColumns_Test = data_test.dtypes[data_test.dtypes == np.object]

# list of columns whose data type is object i.e. string

#print(filteredColumns_Test)

###all_Columns_Object= list(filteredColumns_Test.index) + list(filteredColumns_Train.index)

###print(all_Columns_Object)

#listOfColumnNames = list(all_Columns_Object.index)

#print(listOfColumnNames)

###data_train.drop(all_Columns_Object, axis=1,inplace=True)

###data_test.drop(all_Columns_Object, axis=1,inplace=True)
cols_with_missing = [col for col in data_train.columns if data_train[col].isnull().any()]



# Fill in the lines below: drop columns in training and validation data

reduced_X_train = data_train.drop(cols_with_missing, axis=1,inplace=True)

reduced_X_valid = data_test.drop(cols_with_missing, axis=1,inplace=True)
#print(cols_with_missing)
set(data_train) - set(data_test)
SalePrice = data_train['SalePrice']
data_train.drop(columns='SalePrice',axis=1,inplace=True)

data_test.tail()
from sklearn.impute import SimpleImputer



# Imputation

my_imputer = SimpleImputer(strategy='most_frequent')

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(data_train))

imputed_X_valid = pd.DataFrame(my_imputer.transform(data_test))



# Imputation removed column names; put them back

imputed_X_train.columns = data_train.columns

imputed_X_valid.columns = data_test.columns



data_train = imputed_X_train

data_test = imputed_X_valid
data_test.tail()
s = (data_train.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
##unique_Columns = list()

##for i in object_cols:

 ##   if len(data_train[i].unique()) > 20:

           #print(i ,': ',len(data_train[i].unique()))

   ##         unique_Columns.append(i)

##data_train.drop(columns=unique_Columns,axis=1,inplace=True)

##data_test.drop(columns=unique_Columns,axis=1,inplace=True)
s = (data_train.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
for i in object_cols:

    if len(data_train[i].unique()) > 20:

           print(i ,': ',len(data_train[i].unique()))
set(data_train) - set(data_test)

data_test.tail()
from sklearn.preprocessing import LabelEncoder



# Make copy to avoid changing original data 

label_X_train = data_train.copy()

label_X_valid = data_test.copy()



# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    label_encoder.fit(pd.concat([data_train[col], data_test[col]], axis=0, sort=False))

    label_X_train[col] = label_encoder.transform(data_train[col])

    label_X_valid[col] = label_encoder.transform(data_test[col])

data_train = label_X_train

data_test = label_X_valid

#label_X_train

#print(len(OH_X_train.columns))

#print(len(data_train.columns))

data_test
set(data_train) - set(data_test)
#counter = 0

columns_have_missing_Train = []

for i in data_train.columns:

    if data_train[i].isnull().sum() > 0:

        #counter = counter  + 1

        columns_have_missing_Train.append(i)

        print(i,': ',data_train[i].isnull().sum())

##################################

#counter = 0

columns_have_missing_Test = []

for i in data_test.columns:

    if data_test[i].isnull().sum() > 0:

        #counter = counter  + 1

        columns_have_missing_Test.append(i)

        print(i,': ',data_test[i].isnull().sum())
###setColumne = columns_have_missing_Test + columns_have_missing_Train

###set(setColumne)
###for i in setColumne:

   ### data_train[i].fillna(data_train[i].mean(), inplace=True)

    ###data_test[i].fillna(data_test[i].mean(), inplace=True)   
###data_train
#SalePrice = data_train['SalePrice']

#SalePrice
#data_train.drop(columns='SalePrice',axis=1,inplace=True)
#from sklearn.impute import SimpleImputer



# Imputation

##my_imputer = SimpleImputer()

##imputed_X_train = pd.DataFrame(my_imputer.fit_transform(data_train))

##imputed_X_test = pd.DataFrame(my_imputer.transform(data_test))



# Imputation removed column names; put them back

##imputed_X_train.columns = data_train.columns

##imputed_X_test.columns = data_test.columns

#############################################

##data_train = imputed_X_train

##data_test = imputed_X_test
##SalePrice.index -= 1

##SalePrice

##data_train['SalePrice'] = SalePrice

#data_train
###cols_with_missing_train = [col for col in data_train.columns

   ###                  if data_train[col].isnull().any()]

###cols_with_missing_test = [col for col in data_test.columns

   ###                  if data_test[col].isnull().any()]

#print(cols_with_missing_train)

#print('----------------------')

#print(cols_with_missing_test)

#print(set(cols_with_missing_test) - set(cols_with_missing_train))



###all_missing_columns = cols_with_missing_train + cols_with_missing_test

###print(len(all_missing_columns))



#Drop columns in training and validation data



###data_train.drop(all_missing_columns, axis=1,inplace=True)

###data_test.drop(all_missing_columns, axis=1,inplace=True)



#set(data_test)-set(data_train)
# Get  columns whose data type is object i.e. string

###filteredColumns = data_train.dtypes[data_train.dtypes == np.object]

# list of columns whose data type is object i.e. string

#print(filteredColumns.index)

###listOfColumnNames = list(filteredColumns.index)

###print(listOfColumnNames)

###data_train.drop(listOfColumnNames, axis=1,inplace=True)

###data_test.drop(listOfColumnNames, axis=1,inplace=True)
###data_train
#for i in data_train.columns:    

 #   print(i ,': ',len(data_train[i].unique()))

#len(data_train.Name.unique)
#data_train.drop(columns='SalePrice',axis=1,inplace=True)
#data_train.join(SalePrice = list(SalePrice))
#SalePrice

data_train.insert(0, 'SalePrice',  list(SalePrice))
data_train.SalePrice

#data_train
y = data_train.SalePrice

#############################

X = data_train.drop(columns=['SalePrice'])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error ,explained_variance_score, mean_squared_error
parameters = {'learning_rate':  [0.02,0.05,0.07,0.09], #so called `eta` value

              'max_depth':  list(range(6, 30, 10)),

              'n_estimators': list(range(100, 1001, 100))}

from sklearn.model_selection import GridSearchCV



gsearch = GridSearchCV(estimator=XGBRegressor(),

                       param_grid = parameters, 

                       scoring='neg_mean_absolute_error',

                       n_jobs=4,cv=5,verbose=7)



gsearch.fit(X_train, y_train)



print(gsearch.best_params_.get('n_estimators'))

print(gsearch.best_params_.get('learning_rate'))

print(gsearch.best_params_.get('max_depth'))

#print(gsearch.best_params_.get('subsample'))
parameters_final = {'learning_rate': gsearch.best_params_.get('learning_rate'), #so called `eta` value

              'max_depth': gsearch.best_params_.get('max_depth'),

              'n_estimators': gsearch.best_params_.get('n_estimators')}
my_model = XGBRegressor(learning_rate = gsearch.best_params_.get('learning_rate'),

                         max_depth = gsearch.best_params_.get('max_depth'),

              n_estimators = gsearch.best_params_.get('n_estimators'),random_state=1, n_jobs=4)

my_model.fit(X_train, y_train)

predictions = my_model.predict(X_test)

mean_Error = mean_absolute_error(y_true=y_test,y_pred = predictions)

print(mean_Error)
def getBestScore(n_est):

    my_model = XGBRegressor(n_estimators=n_est,random_state=1,learning_rate=0.05, n_jobs=4)

    my_model.fit(X_train, y_train)

    predictions = my_model.predict(X_test)

    mean_Error = mean_absolute_error(y_true=y_test,y_pred = predictions)

    return mean_Error 

#explained_variance_score

###range_Estimation = getBestScore(1)

###minEstim = 1

###for i in range(1,100,1):

    #print(getBestScore(i),'*-*',i)

   ### if range_Estimation > getBestScore(i):

      ###  minEstim = i

###print(range_Estimation,'>>>',minEstim)

##### 196 is the best...'''
final_model = XGBRegressor(learning_rate = gsearch.best_params_.get('learning_rate'),

                         max_depth = gsearch.best_params_.get('max_depth'),

              n_estimators = gsearch.best_params_.get('n_estimators'),random_state=1, n_jobs=4)

final_model.fit(X, y)

predictions = final_model.predict(X)

#print(predictions)

#mean_absolute_error(y_true=y , y_pred = predictions)

#print(predictions[:5])

#print(y[:5])
data_test
test_preds = final_model.predict(data_test)

#output = pd.DataFrame({'Id': data_test.index,

 #                      'SalePrice': test_preds})

#output.to_csv('submission.csv', index=False)

#############################################3

samplesubmission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

output = pd.DataFrame({'Id': samplesubmission.Id, 'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)

print('Done')