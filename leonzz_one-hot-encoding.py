# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

sample_submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

sales_train_validation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

sales_train_evaluation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
#Do one dataset at a time, and loop thru. Only use the calendar
calendar
data = sales_train_validation[sales_train_validation['id'] == 'HOBBIES_1_001_CA_1_validation'].transpose()[6:].reset_index()

data = data.rename(columns={0:"sales"})

data['moving_sales'] = data['sales'].rolling(window=28).mean()

data['wday'] = calendar['wday'][0:1913]

data = data.drop(columns= ['index'])

data = data.fillna(0)
sales_train_validation
sales_train_evaluation
sales_data = pd.DataFrame(sales_train_validation).reset_index()

for i in range(1914, 1942):

    sales_data['d_' + str(i)] = 0

sales_data
sales_data['state_id']
labels = calendar[['d', 'event_name_1', 'event_name_2', 'snap_CA', 'snap_TX', 'snap_WI', 'wday', 'month', 'year']].reset_index()

labels = labels.fillna(0)

labels = labels.applymap(str)
from sklearn.preprocessing import LabelEncoder

label_encoder=LabelEncoder()

label_encoder = label_encoder.fit(labels['event_name_1'])

label_encoded_event1 = label_encoder.transform(labels['event_name_1'])

label_encoder = label_encoder.fit(labels['event_name_2'])

label_encoded_event2 = label_encoder.transform(labels['event_name_2'])

labels['event_name_1_encode'] = label_encoded_event1

labels['event_name_2_encode'] = label_encoded_event2
#xgb Model

from xgboost import XGBRegressor

model = XGBRegressor(

    max_depth=8,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42)
results = []

ss = pd.DataFrame(sample_submission)
ss
import time 

from tqdm import tqdm
for i in tqdm(range(0, len(sales_data))):

    temp = pd.DataFrame(sales_data.loc[i][7:]).reset_index()

    temp=temp.rename(columns = {"index": "d", i: "sales"})

    sales_all = labels.merge(temp, on=["d"])

    if sales_data['state_id'][i] == 'CA':

        sales_all = sales_all[['event_name_1_encode', 'event_name_2_encode', 'snap_CA', 'wday','month', 'year', 'sales']].applymap(float)

        X_train = sales_all[['event_name_1_encode', 'event_name_2_encode', 'snap_CA', 'wday','month', 'year']][0:1912]

        X_test = sales_all[['event_name_1_encode', 'event_name_2_encode', 'snap_CA', 'wday','month', 'year']][1913:]

    elif sales_data['state_id'][i] == 'TX':

        sales_all = sales_all[['event_name_1_encode', 'event_name_2_encode', 'snap_TX', 'wday','month', 'year', 'sales']].applymap(float)

        X_train = sales_all[['event_name_1_encode', 'event_name_2_encode', 'snap_TX', 'wday','month', 'year']][0:1912]

        X_test = sales_all[['event_name_1_encode', 'event_name_2_encode', 'snap_TX', 'wday','month', 'year']][1913:]

    else: 

        sales_all = sales_all[['event_name_1_encode', 'event_name_2_encode', 'snap_WI', 'wday','month', 'year', 'sales']].applymap(float)

        X_train = sales_all[['event_name_1_encode', 'event_name_2_encode', 'snap_WI', 'wday','month', 'year']][0:1912]

        X_test = sales_all[['event_name_1_encode', 'event_name_2_encode', 'snap_WI', 'wday','month', 'year']][1913:]

    Y_train = sales_all[['sales']][0:1912] 

    Y_test = sales_all['sales'][1913:]

    model.fit(X_train, Y_train)

    y_pred = model.predict(X_test)

    ss.loc[i, 1:] = y_pred
ss.to_csv("submission.csv", index=False)
