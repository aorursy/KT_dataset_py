import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_train
df_train.columns
df_train['SalePrice'].describe()
#пропущенные данные (количество и в процентах)

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
# отбрасываем ненужные данные

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)



df_test = df_test.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_test = df_test.drop(df_test.loc[df_test['Electrical'].isnull()].index)
df_train
def OHE(columns):

    df_final=df_train

    i=0

    for fields in columns:

        df1=pd.get_dummies(df_train[fields],drop_first=True)

        

        df_train.drop([fields],axis=1,inplace=True)

        if i==0:

            df_final=df1.copy()

        else:           

            df_final=pd.concat([df_final,df1],axis=1)

        i=i+1

       

        

    df_final=pd.concat([df_train,df_final],axis=1)

        

    return df_final
def OHE_test(columns):

    df_final1=df_test

    i=0

    for fields in columns:

        df1=pd.get_dummies(df_test[fields],drop_first=True)

        

        df_test.drop([fields],axis=1,inplace=True)

        if i==0:

            df_final1=df1.copy()

        else:           

            df_final1=pd.concat([df_final1,df1],axis=1)

        i=i+1

       

        

    df_final1=pd.concat([df_test,df_final1],axis=1)

        

    return df_final1
columns = ['MSZoning', 'Street',

       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 

       'RoofMatl', 'Exterior1st', 'Exterior2nd',

       'ExterQual', 'ExterCond', 'Foundation',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical',

       'KitchenQual', 'Functional', 'PavedDrive', 'SaleType', 'SaleCondition']
df_train = OHE(columns)

df_train =df_train.loc[:,~df_train.columns.duplicated()]

df_train.shape



df_test = OHE_test(columns)

df_test =df_test.loc[:,~df_test.columns.duplicated()]

df_test.shape
df_test
df_train = df_train.select_dtypes(include=['float64', 'int64', 'bool'])

df_test = df_test.select_dtypes(include=['float64', 'int64', 'bool'])

xtrain= df_train.drop('SalePrice', axis=1)

ytrain= df_train['SalePrice']
xtrain
from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.metrics import accuracy_score

best = 0 

average = 0

total_for_average = 0

model1 = xgb.XGBRegressor(random_state=0,

                        n_estimators=2100, 

                        learning_rate= 0.15,

                        max_depth= 4

                       )

model1.fit(xtrain, ytrain)
#прогноз

preds= model1.predict(df_test)
#вывод

output = pd.DataFrame({'Id': df_test.Id,'SalePrice': preds})

output.to_csv('submission.csv', index=False)