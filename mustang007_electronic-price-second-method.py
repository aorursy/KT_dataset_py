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
train = pd.read_csv('/kaggle/input/ecommerce-data/Train.csv')
test = pd.read_csv('/kaggle/input/ecommerce-data/Test.csv')
df = train

data = pd.concat([df, test], ignore_index=True)
data

# age_dict = {'0-':0, '18-25':1, '26-35':2, '36-45':3, '46-50':4, '51-55':5, '55+':6}
rating_dict = {'0.0-0.5':0, '0.6-1.0':1, '1.1-1.5':2, '1.6-2.0':3, '2.1-2.5':4, '2.6-3.0':5, '3.1-3.5':6, '3.6-4.0':7, '4.1-4.5':8, '4.6-5.0':9}
# train["Age"] = train["Age"].apply(lambda x: age_dict[x])
# test["Age"] = test["Age"].apply(lambda x: age_dict[x])
 
data['Item_Rating'] = data['Item_Rating'].apply(lambda x: np.round(x))
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data = data
data['Product_Brand'] = le.fit_transform(data['Product_Brand'])
data['Item_Category'] = le.fit_transform(data['Item_Category'])
data['Subcategory_1'] = le.fit_transform(data['Subcategory_1'])
data['Subcategory_2'] = le.fit_transform(data['Subcategory_2'])

data.head(5)
data['Item_Rating'] = data['Item_Rating'].astype("O")
data.drop(columns=['Date','Product'], inplace=True)
data.iloc[2451]

train

data.drop(columns='Item_Rating', inplace=True)
data
train_data = data.iloc[:2452]
train_data.tail(4)
test_data = data.iloc[2452:]
train_data
x_data = train_data.iloc[:,:4]
y_data = train_data.iloc[:,4:]

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.1)
from catboost import CatBoostRegressor

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.05, random_state=45)
pipe = make_pipeline(MinMaxScaler(), XGBRegressor())
pipe.fit(X_train, Y_train)
preds = pipe.predict(X_test)
preds = np.abs(preds)
from sklearn.metrics import mean_squared_log_error, r2_score
score = np.sqrt(mean_squared_log_error(Y_test, preds))
print(score)
r2 = r2_score(Y_test, preds)
print("R2 Score:", r2)

test_data = test_data.iloc[:,:4]
test_data
preds = pipe.predict(test_data)
preds = np.abs(preds)
# for i in preds:
#     print(i)