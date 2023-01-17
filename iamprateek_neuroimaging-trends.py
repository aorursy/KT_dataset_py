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
BASE_PATH = '/kaggle/input/trends-assessment-prediction'



# image and mask directories

train_data_dir = f'{BASE_PATH}/fMRI_train'

test_data_dir = f'{BASE_PATH}/fMRI_test'





print('Reading data...')

loading_data = pd.read_csv(f'{BASE_PATH}/loading.csv')

train_data = pd.read_csv(f'{BASE_PATH}/train_scores.csv')

sample_submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')

print('Reading data completed')
train_data.shape
train_data.head()
loading_data.shape
loading_data.head()
# checking missing data

total = train_data.isnull().sum().sort_values(ascending = False)

percent = (train_data.isnull().sum()/train_data.isnull().count()*100).sort_values(ascending = False)

missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_train_data.head()
total = loading_data.isnull().sum().sort_values(ascending = False)

percent = (loading_data.isnull().sum()/loading_data.isnull().count()*100).sort_values(ascending = False)

missing_loading_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_loading_data.head()
BASE_PATH = '/kaggle/input/trends-assessment-prediction'



fnc_df = pd.read_csv(f"{BASE_PATH}/fnc.csv")

loading_df = pd.read_csv(f"{BASE_PATH}/loading.csv")

labels_df = pd.read_csv(f"{BASE_PATH}/train_scores.csv")
fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])

df = fnc_df.merge(loading_df, on="Id")

labels_df["is_train"] = True

df = df.merge(labels_df, on="Id", how="left")

test_df = df[df["is_train"] != True].copy()

train_df = df[df["is_train"] == True].copy()
target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']

y_train_df = train_df[target_cols]

test_df = test_df.drop(target_cols + ['is_train'], axis=1)
train_df = train_df.drop(target_cols + ['is_train'], axis=1)
df.info()
FNC_SCALE = 1/500

test_df[fnc_features] *= FNC_SCALE

train_df[fnc_features] *= FNC_SCALE
def metric(y_true, y_pred):

    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))
from sklearn.model_selection import KFold, train_test_split

import lightgbm as lgb
param = {'objective':'regression',

        'metric':'rmse',

        'bossting_type':'gbdt',

        'learning_rate':0.01,

        'max_depth':-1}



output = pd.DataFrame()



for target in ['age','domain1_var1','domain1_var2','domain2_var1','domain2_var2']:

    

    X_train, X_val, y_train, y_val = train_test_split(train_df.iloc[:,1:], y_train_df[target], test_size=0.2, shuffle=True, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train)

    val_data = lgb.Dataset(X_val, label=y_val)

    

    model = lgb.train(param, 

                      train_data, 

                      10000, 

                      early_stopping_rounds=15, 

                      valid_sets=[val_data], 

                      verbose_eval=50)

    

    temp = pd.DataFrame(test_df['Id'].apply(lambda x:str(x)+ '_'+ target))

    temp['Predicted'] = model.predict(test_df.iloc[:,1:])

    output = pd.concat([output,temp])
sample_submission = pd.read_csv("/kaggle/input/trends-assessment-prediction/sample_submission.csv")

output = sample_submission.drop('Predicted',axis=1).merge(output,on='Id',how='left')

output.to_csv("submission.csv", index=False)