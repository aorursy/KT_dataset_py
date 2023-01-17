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
# Importing Libraries

import pandas as pd

import numpy as np

from datetime import datetime

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error

from math import sqrt
# Loading training 

train = pd.read_csv("/kaggle/input/into-the-future/train.csv")

train.head()
# loading testing data 

test = pd.read_csv("/kaggle/input/into-the-future/test.csv")

test.head()
# changed datatype of time column is changes from a string to date_time object

date = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')

train = pd.read_csv("/kaggle/input/into-the-future/train.csv", parse_dates=['time'], date_parser=date)

test = pd.read_csv("/kaggle/input/into-the-future/test.csv", parse_dates=['time'], date_parser=date)

train.info()
test.info()
# Converting datetime object into float

def datetime_to_float(d):

    return d.timestamp()



for i in range(len(train["time"])):

    train.loc[i, "time"]=datetime_to_float(train.loc[i, "time"])

train["time"]=train.time.astype(float)

train.info()
def datetime_to_float(d):

    return d.timestamp()



for i in range(len(test["time"])):

    test.loc[i, "time"]=datetime_to_float(test.loc[i, "time"])

test["time"]=test.time.astype(float)

test.info()
test.head()
x=train.loc[:, ["time"]]

y=train["feature_2"]

X_train, x_test, Y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(x_test)
print("RMSE Score:", sqrt(mean_squared_error(y_test, y_pred)))
id = test["id"]
# seperating dependent and independent variablea

test=test.loc[:, ["time"]]
predictions = regressor.predict(test)

predictions
results=pd.DataFrame({"id": id, "feature_2":predictions})

results
results.to_csv("/kaggle/working/solution1.csv", index=False)