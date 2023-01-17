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
train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")

test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")
country_list = train_df['Country/Region'].unique()

number_list = list(range(len(country_list)))

country_num_map = dict(zip(country_list, number_list))
train_df['Country_id'] = train_df['Country/Region'].map(country_num_map)

test_df['Country_id'] = test_df['Country/Region'].map(country_num_map)
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
# test_date_start = test_df['Date'].min()

# train_df = train_df[train_df['Date'] < test_date_start]
pop_df = pd.read_csv('/kaggle/input/additional/population.csv')

gdp_df = pd.read_csv('/kaggle/input/additional/GDP.csv')

age_df = pd.read_csv('/kaggle/input/additional/median_age.csv')

hosbed_df = pd.read_csv('/kaggle/input/additional/hospital_beds.csv')

deathrate_df = pd.read_csv('/kaggle/input/additional/death_rate.csv')
pop_df = pop_df.rename(columns={'name': 'Country', 'pop2019': 'Population', 'area': 'Area'})

pop_df = pop_df[['Country', 'Population', 'Area', 'Density']]



gdp_df = gdp_df.rename(columns={'Country Name': 'Country', '2018': 'GDP'})



age_df = age_df.rename(columns={'Place': 'Country', 'Median': 'Median_Age'})

age_df = age_df[['Country', 'Median_Age']]



hosbed_df = hosbed_df.rename(columns={'Country Name': 'Country', 'bed': 'Hospital_Beds'})

hosbed_df = hosbed_df[['Country', 'Hospital_Beds']]



deathrate_df = deathrate_df.rename(columns={'name': 'Country', 'Rate': 'Death_Rate'})
for df in [pop_df, gdp_df, age_df, hosbed_df, deathrate_df]:

    train_df = train_df.merge(df, left_on='Country/Region', right_on='Country', how='left')
train_df = train_df.drop(columns=['Id', 'Province/State', 'Country_x', 'Country_y',

                                          'Country_x', 'Country_y', 'Country'])
train_df = train_df.fillna(train_df.mean())
for col in train_df.columns:

    count_na = train_df[col].isna().sum()

    fill_rate = 1 - count_na / len(train_df)

    print(col + ': ' + str(fill_rate))
df_tmp = train_df.copy()

df_tmp = df_tmp.drop_duplicates(subset='Country/Region', keep='first')

df_tmp = df_tmp[['Country/Region', 'Population', 'Area', 'Density', 'GDP', 'Median_Age', 'Hospital_Beds',

       'Death_Rate']]

test_df = test_df.merge(df_tmp, on='Country/Region', how='left')
cols_1 = ['Country_id', 'Lat', 'Long', 'Date_day', 'Date_month', 'Date_weekofyear', 'Date_dayofyear']

cols_2 = ['Country_id', 'Lat', 'Long', 'Date_day', 'Date_month', 'Date_weekofyear', 'Date_dayofyear', 

          'Population', 'Density', 'GDP', 'Median_Age', 'Hospital_Beds', 'Death_Rate']
x1 = train_df[cols_2]

x2 = train_df[cols_2]

y1 = train_df['ConfirmedCases']

y2 = train_df['Fatalities']

x_test_1 = test_df[cols_2]

x_test_2 = test_df[cols_2]
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
dt_r = DecisionTreeRegressor(random_state = 0) 

rf = RandomForestRegressor(random_state = 0) 
dt_r.fit(x1,y1)

pred1 = dt_r.predict(x_test_1)

pred1 = pd.DataFrame(pred1)

pred1.columns = ["ConfirmedCases_prediction"]
dt_r.fit(x2,y2)

pred2 = dt_r.predict(x_test_2)

pred2 = pd.DataFrame(pred2)

pred2.columns = ["Death_prediction"]
sub_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/submission.csv")
sub_new_df = sub_df[["ForecastId"]]

OP = pd.concat([pred1,pred2,sub_new_df],axis=1)

OP.columns = ['ConfirmedCases', 'Fatalities', 'ForecastId']

OP = OP[['ForecastId','ConfirmedCases', 'Fatalities']]

OP["ConfirmedCases"] = OP["ConfirmedCases"].astype(int)

OP["Fatalities"] = OP["Fatalities"].astype(int)
OP.to_csv("submission.csv",index=False)