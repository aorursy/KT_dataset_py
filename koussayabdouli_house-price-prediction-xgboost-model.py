import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



### Import Data

train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test_data  = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
print("train_data shape" , train_data.shape)

print("test_data shape" , test_data.shape)
train_data['train_set'] = 1

test_data['train_set'] = 0

data = pd.concat([train_data , test_data] , axis = 0 )
data.head()
print("Data Shape : train data + test data is " , data.shape)
sns.heatmap(data.isna())
# Check the missing values

def Missing_values(df):

    list_missing_values = []

    for col in data:

        percentage = (df[col].isna().sum()/ df.shape[0])*100

        list_missing_values.append((col , percentage))

    Missing_values = pd.DataFrame(list_missing_values , columns = ['Name' , 'Percentage %'])

    return Missing_values
MISS_VAL = Missing_values(data)

MISS_VAL[MISS_VAL['Percentage %'] > 70 ]
# Drop all columns have more than 70 % of missing values .

data_edit = data.drop(['Id' ,'Alley' , 'PoolQC' , 'Fence' , 'MiscFeature' ] , axis = 1)
data_edit.head()
print(data_edit.shape)
# % of value types 

print(data.dtypes.value_counts())

data.dtypes.value_counts().plot.pie()
def Get_Categorical_features(df):

    for col in data.select_dtypes('object'):

        print(f'{col :-<10} {data[col].unique()}')

# Here we can get the unique values of each column

Get_Categorical_features(data_edit)
data_categorical = data_edit.select_dtypes('object')
print("Shape of Categorical data is :",data_categorical.shape )

data_categorical.head()
for col in data_categorical:

    plt.figure()

    sns.countplot(y = col , data = data_categorical)

    plt.title(col)

    
data_categorical.isna().sum()
def Fill_missing_values(data):

    for col in data:

        data[col] = data[col].fillna(data[col].mode()[0])

                                     

    

            
Fill_missing_values(data_categorical)

data_categorical.isna().sum()
data_numerical = data.select_dtypes(exclude=['object'])

data_numerical.head()
# Target variable :

sns.distplot(data_numerical['SalePrice'])
data_numerical.isna().sum()
data_numerical['LotFrontage'].describe()
# We fill values of LotFrontage with 68

data_numerical['LotFrontage'] = data_numerical['LotFrontage'].fillna(68)
# We fill values of GarageYrBlt using this way :

diff = (data_numerical['YrSold'] - data_numerical['YearBuilt']).median()

data_numerical['GarageYrBlt'] = data_numerical['GarageYrBlt'].fillna(data_numerical['YrSold'] - diff)
for col in data_numerical:

    data_numerical[col] = data_numerical[col].fillna(data_numerical[col].mode()[0])
data_numerical.loc[: , ['YearBuilt'  , 'YearRemodAdd' , 'YrSold' ]]

data_numerical['House_Age'] = data_numerical['YrSold'] - data_numerical['YearBuilt']

data_numerical['Remod_First_Age'] = data_numerical['YearRemodAdd'] - data_numerical['YearBuilt']
def State_Feature(Remod_First_Age) : 

    if (Remod_First_Age < 10):

        return 3

    elif (Remod_First_Age >= 10) & (Remod_First_Age < 30):

        return 2

    elif (Remod_First_Age >= 30) & (Remod_First_Age < 100):

        return 1

    else:

        return 0

data_numerical['State'] = data_numerical['Remod_First_Age'].apply(lambda x : State_Feature(x))
data_numerical[data_numerical['House_Age'] < 0] 

print(data_numerical[data_numerical['House_Age'] < 0]['YrSold'])

print(data_numerical[data_numerical['House_Age'] < 0]['YearBuilt'])
data_numerical.loc[data_numerical['House_Age'] < 0, ['YrSold']] = 2008
data_categorica_encoding = pd.get_dummies(data_categorical , drop_first= True)
data_final = pd.concat([data_categorica_encoding , data_numerical] , axis = 1 , sort = False)
# divide the data_final to train and test data :

train_data_final = data_final[data_final['train_set'] == 1]

test_data_final = data_final[data_final['train_set'] == 0]
train_data_final.head()

y = train_data_final['SalePrice']

train_data_final.drop(['SalePrice' , 'train_set'] , axis = 1 , inplace = True)

test_data_final.drop(['SalePrice' , 'train_set'] , axis = 1 , inplace = True)

import xgboost as xgb

from sklearn.model_selection import train_test_split , RandomizedSearchCV 
model = xgb.XGBRegressor()

model
booster=['gbtree','gblinear']

base_score=[0.25,0.5,0.75,1 ,1.25]

max_depth = [2, 3 , 5 ,7 , 9 , 11 ,15]

n_estimators = [100, 500, 800, 1200, 1500]

learning_rate=[0.05,0.1,0.15,0.20 , 0.25 , 0.30]

min_child_weight=[1,2,3,4,5]



params_grid = {

    'booster' : booster ,

    'base_score' : base_score , 

    'max_depth' : max_depth , 

    'n_estimators' : n_estimators ,

    'learning_rate' : learning_rate , 

    'min_child_weight' : min_child_weight

}

RSCV_model = RandomizedSearchCV(

            estimator=model,

            param_distributions=params_grid,

            cv=5, n_iter=60,

            scoring = 'neg_mean_absolute_error',

            n_jobs = 4,

            verbose = 5, 

            return_train_score = True,

            random_state=42 )
X_train , X_test , y_train , y_test = train_test_split(train_data_final.values , y.values , test_size = 0.3 ,

                                                       random_state = 42)
RSCV_model.fit(X_train , y_train )
RSCV_model.best_estimator_
XGB_Regressor = RSCV_model.best_estimator_

XGB_Regressor.fit(X_train , y_train)
test_data_final
y_pred = XGB_Regressor.predict(test_data_final.values)
Submission = pd.DataFrame(

{

    "ID" : test_data_final['Id'] ,

    'SalePrice' : y_pred

})
Submission.to_csv('Submission.csv' , index = False)