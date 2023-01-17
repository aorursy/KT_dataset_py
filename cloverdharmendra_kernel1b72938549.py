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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

# Load the dataset

df = pd.read_csv(r"/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

#first few rows

df.head()
# Find out missing values

null_value = df.isnull().sum()

# print("Null value into dataset ", null_value)



# fill nan value by 0

df = df.fillna(value=0)





# convert column into lower case



for item in df.columns:

    try:

        df[item] = df[item].str.lower()

    except :

        print("Can't convert to lower case")





# Convert yes and no to 0, 1

columns_to_convert = ['Partner',

                      'Dependents',

                      'PhoneService',

                      'PaperlessBilling',

                      'Churn']



for item in columns_to_convert:

    df[item].replace(to_replace="yes", value=1, inplace=True)

    df[item].replace(to_replace="no", value=0, inplace=True)



# print(df.head(100))

df['TotalCharges'] = df['TotalCharges'].replace(r'\s+', np.nan, regex=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
# remove customer id

try:

    customer_id = df['customerID']

    del df['customerID']

except:

    print("already removed id")

# using One-hot encoding, Converting str to integer

df  =pd.get_dummies(df)

df.fillna(value=0, inplace=True)





# spilt data to lable and non label



try:

    label = df['Churn'] # Remove the label before training the model

    del df['Churn']

except:

    print("label already removed.")
df.head(10)
# change data to StandardScaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# df = scaler.fit(df)
df = scaler.fit_transform(df)

X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.2, random_state=123)



# Now create classifier

xg_reg = xgb.XGBClassifier(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.0001,

                            max_depth = 15, alpha = 10, n_estimators = 5)



xg_reg.fit(df, label)



preds = xg_reg.predict(X_test)



rmse = np.sqrt(mean_squared_error(y_test, preds))

score = xg_reg.score(X_test, y_test)
score