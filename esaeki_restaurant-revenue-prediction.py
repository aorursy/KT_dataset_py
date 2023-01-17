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
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
df_train = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/train.csv.zip')
df_test = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv.zip')
df_train.head()
df_train.isnull().sum()
df_test.head()
df_test.isnull().sum()
revenue = df_train["revenue"]
del df_train["revenue"]
df_whole = pd.concat([df_train, df_test], axis=0)
df_whole.info()
#Created "BusinessPeriod" column
df_whole["Open Date"] = pd.to_datetime(df_whole["Open Date"])
df_whole["Year"] = df_whole["Open Date"].apply(lambda x:x.year)
df_whole["Month"] = df_whole["Open Date"].apply(lambda x:x.month)
df_whole["Day"] = df_whole["Open Date"].apply(lambda x:x.day)

df_whole["from"] = "2015-04-27"
df_whole["from"] = pd.to_datetime(df_whole["from"])
df_whole["BusinessPeriod"] = (df_whole["from"] - df_whole["Open Date"]).apply(lambda x: x.days)

df_whole = df_whole.drop('Open Date', axis=1)
df_whole = df_whole.drop('from', axis=1)
le = LabelEncoder()
df_whole["City"] = le.fit_transform(df_whole["City"])
df_whole["City Group"] = df_whole["City Group"].map({"Other":0, "Big Cities":1})
df_whole["Type"] = df_whole["Type"].map({"FC":0, "IL":1, "DT":2, "MB":3})
df_train = df_whole.iloc[:df_train.shape[0]]
df_test = df_whole.iloc[df_train.shape[0]:]
#Random Forest model
df_train_columns = [col for col in df_train.columns if col not in ["Id"]]

rf = RandomForestRegressor(
    n_estimators=200, 
    max_depth=5, 
    max_features=0.5, 
    random_state=449,
    n_jobs=-1
)
rf.fit(df_train[df_train_columns], revenue)
prediction = rf.predict(df_test[df_train_columns])
submission = pd.DataFrame({"Id":df_test.Id, "Prediction":prediction})
submission.to_csv("submission3.csv", index=False)