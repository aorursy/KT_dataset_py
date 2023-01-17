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
test = pd.read_csv("../input/big-mart-sales-prediction/test_AbJTz2l.csv")

submission = pd.read_csv("../input/big-mart-sales-prediction/sample_submission_8RXa3c6.csv")

train = pd.read_csv("../input/big-mart-sales-prediction/train_v9rqX0R.csv")
test
test = test.merge(submission, on=['Item_Identifier','Outlet_Identifier'])

test
train
train_test = pd.concat([train,test], ignore_index=True)

train_test
non_number_train_test_columns = train_test.dtypes[train_test.dtypes == object].index.values

for columns in non_number_train_test_columns:

    print(columns)

    print(train_test[columns].value_counts())

    print()



non_number_train_test_columns
train_test.isnull().sum()
train_test.isnull().sum()
train_test.loc[train_test['Item_Fat_Content'] == 'reg', 'Item_Fat_Content'] = 'Regular'

train_test.loc[train_test['Item_Fat_Content'] == 'LF', 'Item_Fat_Content'] = 'Low Fat'

train_test.loc[train_test['Item_Fat_Content'] == 'low fat', 'Item_Fat_Content'] = 'Low Fat'



train_test.loc[train_test['Outlet_Size'] == 'High', 'Outlet_Size'] = 3

train_test.loc[train_test['Outlet_Size'] == 'Medium', 'Outlet_Size'] = 2

train_test.loc[train_test['Outlet_Size'] == 'Small', 'Outlet_Size'] = 1



current = ['Tier '+str(i) for i in range(1,4)]

update = [1,2,3]

for i in range(len(update)):

    train_test.loc[train_test['Outlet_Location_Type'] == current[i], 'Outlet_Location_Type'] = update[i]



current = ['Supermarket Type'+str(i) for i in range(1,4)]+['Grocery Store']

update = [1,2,3,4]

for i in range(len(update)):

    train_test.loc[train_test['Outlet_Type'] == current[i], 'Outlet_Type'] = update[i]



train_test.loc[train_test['Item_Fat_Content'] == 'Low Fat', 'Item_Fat_Content'] = 2

train_test.loc[train_test['Item_Fat_Content'] == 'Regular', 'Item_Fat_Content'] = 1    



for columns in non_number_train_test_columns:

    print(columns)

    print(train_test[columns].value_counts())

    print()
values = {'Item_Weight': 0, 'Outlet_Size': 0}

train_test = train_test.fillna(value=values)



train_test['Item_Fat_Content'] = train_test['Item_Fat_Content'].astype(int)

train_test['Outlet_Size'] = train_test['Outlet_Size'].astype(int)

train_test['Outlet_Location_Type'] = train_test['Outlet_Location_Type'].astype(int)

train_test['Outlet_Type'] = train_test['Outlet_Type'].astype(int)

train_test['Item_Fat_Content'] = train_test['Item_Fat_Content'].astype(int)



non_number_train_test_columns = train_test.dtypes[train_test.dtypes == object].index.values

for columns in non_number_train_test_columns:

    print(columns)

    print(train_test[columns].value_counts())

    print()

    

non_number_train_test_columns
from sklearn.preprocessing import LabelEncoder

for column in non_number_train_test_columns:

    le = LabelEncoder()

    train_test[column] = le.fit_transform(train_test[column]).astype(np.int64)
train_test
def train_test_split(train,test,train_test,remove_column,y_value,ratio):

    x_train = train_test.iloc[:len(train)*10*ratio//10].drop(remove_column, axis=1)

    x_val = train_test.iloc[len(train)*10*ratio//10:].drop(remove_column, axis=1)



    y_train = train_test.iloc[:len(train)*10*ratio//10][y_value]

    y_val = train_test.iloc[len(train)*10*ratio//10:][y_value]

    return x_train, x_val, y_train, y_val
x_train = train_test.iloc[:len(train)*9//10].drop(['Item_Outlet_Sales'], axis=1)

x_val = train_test.iloc[len(train)*9//10:].drop(['Item_Outlet_Sales'], axis=1)



y_train = train_test.iloc[:len(train)*9//10]['Item_Outlet_Sales']

y_val = train_test.iloc[len(train)*9//10:]['Item_Outlet_Sales']
import time

from xgboost import XGBRegressor

ts = time.time()



model = XGBRegressor(

    max_depth=10,

    n_estimators=1000,

    min_child_weight=0.5, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.1,

#     tree_method='gpu_hist',

    seed=42)



model.fit(

    x_train, 

    y_train, 

    eval_metric="rmse", 

    eval_set=[(x_train, y_train), (x_val, y_val)], 

    verbose=True, 

    early_stopping_rounds = 20)



time.time() - ts
x_test = train_test.iloc[len(train):].drop(['Item_Outlet_Sales'], axis=1)



Y_pred = model.predict(x_val)

Y_test = model.predict(x_test)
Y_test
submission['Item_Outlet_Sales'] = Y_test#

submission.to_csv('submission.csv',index=False)

submission
# train_test[train_test['Item_Fat_Content'] == 'reg'] # train_test['Item_Fat_Content'] == 'reg'
# for col in df.columns():

#    df.loc[df[col] == 'n', col] = 0
# train_test['Item_Fat_Content'] == 'reg'
submission