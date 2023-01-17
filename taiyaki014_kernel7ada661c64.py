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
import pandas as pd

df_train = pd.read_csv('../input/restaurant-revenue-prediction/train.csv.zip')

df_test = pd.read_csv('../input/restaurant-revenue-prediction/test.csv.zip')
train_corr = df_train.corr()

print(train_corr)

train_corr.to_csv("corr.csv", index = False)
import datetime

from sklearn.preprocessing import LabelEncoder

revenue = df_train["revenue"]
del df_train["revenue"]
df_whole = pd.concat([df_train, df_test], axis=0)
df_whole["Open Date"] = pd.to_datetime(df_whole["Open Date"])

df_whole["Year"] = df_whole["Open Date"].apply(lambda x:x.year)

df_whole["Month"] = df_whole["Open Date"].apply(lambda x:x.month)

df_whole["Day"] = df_whole["Open Date"].apply(lambda x:x.day)
le = LabelEncoder()

df_whole["City"] = le.fit_transform(df_whole["City"])
df_whole["City Group"] = df_whole["City Group"].map({"Other":0, "Big Cities":1})
df_whole["Type"] = df_whole["Type"].map({"FC":0, "IL":1, "DT":2, "MB":3})
df_train = df_whole.iloc[:df_train.shape[0]]

df_test = df_whole.iloc[df_train.shape[0]:]
from sklearn.ensemble import RandomForestRegressor



#学習に使う特徴量を取得

df_train_columns = [col for col in df_train.columns if col not in ["Id", "Open Date"]]



#RandomForestで学習させる

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

submission.to_csv("TFI_submission.csv", index=False)