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
df_train = pd.read_csv("/kaggle/input/santander-value-prediction-challenge/train.csv")
df_test = pd.read_csv("/kaggle/input/santander-value-prediction-challenge/test.csv")
df_train
#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
dtype_df = df_train.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()
unique_df = df_train.nunique().reset_index()
unique_df.columns = ["col_name", "unique_count"]
constant_df = unique_df[unique_df["unique_count"]==1]
constant_df.shape
df_train.drop(constant_df.col_name.tolist(), axis=1, inplace=True)

# remove constant columns in the test set
df_test.drop(constant_df.col_name.tolist(), axis=1, inplace=True) 

print("Removed `{}` Constant Columns\n".format(len(constant_df.col_name.tolist())))
print(constant_df.col_name.tolist())
%%time
def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break

    return dups

colsToRemove = duplicate_columns(df_train)
print(colsToRemove)
df_train.drop(colsToRemove, axis=1, inplace=True) 

# remove duplicate columns in the testing set
df_test.drop(colsToRemove, axis=1, inplace=True)

print("Removed `{}` Duplicate Columns\n".format(len(colsToRemove)))
print(colsToRemove)
df_test
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
featureNames = df_train.columns[1:-1]
df_train
df = df_train.copy(deep=True)
df = df[:1000]
df
xtrain= df.drop('target', axis=1)
ytrain= df['target']
xtrain
df.pop("ID")
df
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
df_test.pop("ID")
df_test
#prediction
preds= model1.predict(df_test)
#output
db=pd.read_csv("/kaggle/input/santander-value-prediction-challenge/sample_submission.csv")
db['target'] = abs(preds)
db.to_csv("submission.csv", index = False)
db