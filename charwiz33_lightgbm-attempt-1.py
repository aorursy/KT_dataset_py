# Archit Kulkarni, Adhithya Narayanan, Vibhu Ambil, Tyler Youngberg

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
'''

Title: M5 Data Preprocessing

Author: Quinn Wang

Date: March 2020

Availability: https://www.kaggle.com/qcw171717/naive-baseline/

'''



import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

from tqdm import tqdm

df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')

price_df = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")

cal_df = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")

cal_df["d"]=cal_df["d"].apply(lambda x: int(x.split("_")[1]))

price_df["id"] = price_df["item_id"] + "_" + price_df["store_id"] + "_validation"

for i in range(1886, 1914):

    df["F_" + str(i)] = 0



'''end Quinn Wang'''
import lightgbm as lgb
temp = np.zeros((30490,), dtype=int)

k = 0

for i in df["state_id"]:

    if i == "CA":

        temp[k] = 0

    if i == "TX":

        temp[k] = 1

    if i == "WI":

        temp[k] = 2

    k+=1

df["state_id"] = pd.DataFrame(temp)
temp = np.zeros((30490,), dtype=int)

k = 0

for i in df["store_id"]:

    if i == "CA_1":

        temp[k] = 0

    elif i == "CA_2":

        temp[k] = 1

    elif i == "CA_3":

        temp[k] = 2

    elif i == "CA_4":

        temp[k] = 3

    elif i == "TX_1":

        temp[k] = 4

    elif i == "TX_2":

        temp[k] = 5

    elif i == "TX_3":

        temp[k] = 6

    elif i == "WI_1":

        temp[k] = 7

    elif i == "WI_2":

        temp[k] = 8

    elif i == "WI_3":

        temp[k] = 9

    k+=1

df["store_id"] = pd.DataFrame(temp)
temp = np.zeros((30490,), dtype=int)

k = 0

for i in df["cat_id"]:

    if i == "HOBBIES":

        temp[k] = 0

    if i == "FOODS":

        temp[k] = 1

    if i == "HOUSEHOLD":

        temp[k] = 2

    k+=1

df["cat_id"] = pd.DataFrame(temp) 
temp = np.zeros((30490,), dtype=int)

k = 0

for i in df["dept_id"]:

    if i == "HOBBIES_1":

        temp[k] = 0

    elif i == "HOBBIES_2":

        temp[k] = 1

    elif i == "FOODS_1":

        temp[k] = 2

    elif i == "FOODS_2":

        temp[k] = 3

    elif i == "FOODS_3":

        temp[k] = 4

    elif i == "HOUSEHOLD_1":

        temp[k] = 5

    elif i == "HOUSEHOLD_2":

        temp[k] = 6

    k+=1

df["dept_id"] = pd.DataFrame(temp)
validation = pd.DataFrame()

train = df

for x in range(1886, 1914):

    s = "d_" + str(x)

    f = "F_" + str(x)

    train = train.drop(columns = [s, f])

    validation[s] = df[s]
train
for i in range(1, 1857):

    d = "d_" + str(i)

    train = train.drop(columns = [d])
train = train.drop(columns=["item_id", "id"])
params = {

   'task': 'train',

   'boosting_type': 'gbdt',

   'objective': 'regression',

   'metric': 'rmse',

   'learning_rate': .05,

   'num_iterations': 50,

}
for x in range(1886, 1914):    

    

    # setup for training

    t = train.values

    v = validation.values[:, x - 1886]

    lgbm_data = lgb.Dataset(t, label=v, 

                            feature_name=train.columns.tolist(), 

                            categorical_feature=['dept_id', 'cat_id', 'store_id', 'state_id'])

    

    # train data

    gbt = lgb.train(params, lgbm_data) 

    

    s = "d_" + str(x) # string name for day in format d_xxxx

    x = pd.DataFrame(gbt.predict(t))

    

    # add new data to training for next iteration

    train[s] = x
forecast = pd.DataFrame(df["id"])

for i in range(1886, 1914):

    d = "d_" + str(i)

    f = "F" + str(i-1885)

    forecast[f] = train[d]
from sklearn.metrics import mean_squared_error as mse

import math

math.sqrt(mse(validation.values, forecast.drop(columns=["id"]).values))
submission_df2 = forecast.copy()

submission_df2["id"] = submission_df2["id"].apply(lambda x : x.replace('validation', 'evaluation'))
forecast = forecast.append(submission_df2).reset_index(drop=True)
forecast
forecast.to_csv("forecast.csv", index=False)