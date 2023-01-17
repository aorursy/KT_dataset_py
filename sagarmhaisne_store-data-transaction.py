# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_wrk = pd.read_csv('/kaggle/input/store-transaction-data/Hackathon_Working_Data.csv')
df_val = pd.read_csv('/kaggle/input/store-transaction-data/Hackathon_Validation_Data.csv')
# Data Preprocessing
# Considering Features
X_train = df_wrk.iloc[:,[0,1,8]]
y_train = df_wrk.iloc[:,[7]]
X_test = df_val.iloc[:,1:4]

# Swapping the columns of validation data
columns_titles = ["MONTH","STORECODE","GRP"]
X_test = X_test.reindex(columns=columns_titles)
# Encoding Month, Store and Group columns

le = LabelEncoder()
X_train = X_train.apply(le.fit_transform)
X_test = X_test.apply(le.fit_transform)
# Applying XgBoost regressor
model = XGBRegressor()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# Prepare Dataframe
data = { 'ID' : pd.Series(df_val['ID']),
        'VALUE' : pd.Series(yhat)
        }
df_fin = pd.DataFrame(data)

print(df_fin)