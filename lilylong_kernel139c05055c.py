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
train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")
country_list = train_df['Country_Region'].unique()

number_list = list(range(len(country_list)))

country_num_map = dict(zip(country_list, number_list))
train_df['Country_id'] = train_df['Country_Region'].map(country_num_map)

test_df['Country_id'] = test_df['Country_Region'].map(country_num_map)
train_df['Province_State'] = np.where(train_df['Province_State'].isna(), train_df['Country_Region'], 

                                     train_df['Province_State'])

test_df['Province_State'] = np.where(test_df['Province_State'].isna(), test_df['Country_Region'], 

                                     test_df['Province_State'])
from datetime import datetime
train_df["Date_date"] = train_df["Date"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

test_df["Date_date"] = test_df["Date"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
train_df['Date_day'] = train_df["Date_date"].apply(lambda x: x.day)

train_df['Date_month'] = train_df["Date_date"].apply(lambda x: x.month)

train_df['Date_weekofyear'] = train_df["Date_date"].apply(lambda x: x.week)

train_df['Date_dayofweek'] = train_df["Date_date"].apply(lambda x: x.dayofweek)

train_df['Date_dayofyear'] = train_df["Date_date"].apply(lambda x: x.dayofyear)
test_df['Date_day'] = test_df["Date_date"].apply(lambda x: x.day)

test_df['Date_month'] = test_df["Date_date"].apply(lambda x: x.month)

test_df['Date_weekofyear'] = test_df["Date_date"].apply(lambda x: x.week)

test_df['Date_dayofweek'] = test_df["Date_date"].apply(lambda x: x.dayofweek)

test_df['Date_dayofyear'] = test_df["Date_date"].apply(lambda x: x.dayofyear)
train_df["Date"] = train_df["Date"].apply(lambda x: x.replace("-",""))

train_df["Date"] = train_df["Date"].astype(int)
test_df["Date"] = test_df["Date"].apply(lambda x: x.replace("-",""))

test_df["Date"]  = test_df["Date"].astype(int)
test_date_start = test_df['Date'].min()

train_df = train_df[train_df['Date'] < test_date_start]
from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline



poly_lr = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)),

                  ('linear', LinearRegression())])



poly_ridge = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)),

                  ('linear', Ridge(random_state=0))])
pred_df=pd.DataFrame(columns=['ConfirmedCases_prediction', 'Death_prediction'])

pred1 = []

pred2 = []



model = poly_ridge

cols_c = ['Date', 'Date_day', 'Date_month', 'Date_weekofyear', 'Date_dayofyear']



for country_id in train_df['Country_id'].unique():

    train_df_c = train_df[train_df['Country_id'] == country_id]

    for state in train_df_c['Province_State'].unique():

        train_df_c = train_df[(train_df['Country_id'] == country_id) &

                              (train_df['Province_State'] == state)]

        test_df_c = test_df[(test_df['Country_id'] == country_id) &

                              (test_df['Province_State'] == state)]

        x_train_c = train_df_c[cols_c]

        y1_train_c = train_df_c['ConfirmedCases']

        y2_train_c = train_df_c['Fatalities']

        x_test_c = test_df_c[cols_c]

        model.fit(x_train_c, y1_train_c)

        pred1_c = model.predict(x_test_c)

        pred1_c = np.where(pred1_c>=0, pred1_c, [0]*len(pred1_c))

        pred1.extend(list(pred1_c))

        pred_df_c = pd.DataFrame(pred1_c)

        pred_df_c.columns = ["ConfirmedCases_prediction"]

        model.fit(x_train_c,y2_train_c)

        pred2_c = model.predict(x_test_c)

        pred2_c = np.where(pred2_c>=0, pred2_c, [0]*len(pred2_c))

        pred2.extend(list(pred2_c))

        pred_df_c["Death_prediction"] = pred2_c

        pred_df = pd.concat([pred_df, pred_df_c], axis=0)
pred_df = pred_df.reset_index(drop=True)

pred_df.head()
sub_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")
sub_new_df = sub_df[["ForecastId"]]

# OP = pd.concat([sub_new_df, pred1, pred2],axis=1)

OP = pd.concat([sub_new_df, pred_df],axis=1)

OP.columns = ['ForecastId', 'ConfirmedCases', 'Fatalities']

OP["ConfirmedCases"] = OP["ConfirmedCases"].astype(int)

OP["Fatalities"] = OP["Fatalities"].astype(int)
OP.to_csv("submission.csv",index=False)