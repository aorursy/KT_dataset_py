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
df=pd.read_csv("/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv")
df.head()
df=df.drop('Unnamed: 0',axis=1)
df.describe(include='all')
df.info(verbose=True)
col_list=list(df)
#excluding the "0" the column, date column, as it will be converted to date-time object.

df[col_list[1:]] = df[col_list[1:]].apply(pd.to_numeric, errors='coerce')
df['Time Serie']=pd.to_datetime(df['Time Serie'])
df.info(verbose=True)
df_melted=df.melt(id_vars=["Time Serie"], 

        var_name="Currency type", 

        value_name="Value")
df_melted.head()
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt



sns.set_style("white")

plt.figure(figsize=(20,10))

sns.lineplot(x="Time Serie", y="Value", hue="Currency type",palette='deep',data=df_melted)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
df_melted_kor=df_melted.loc[df_melted['Currency type'] == "KOREA - WON/US$"]

df_melted_nonkor=df_melted.loc[df_melted['Currency type'] != "KOREA - WON/US$"]
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt



sns.set_style("white")

fig, ax1 = plt.subplots(figsize=(20,10))

ax2 = ax1.twinx()



g=sns.lineplot(x="Time Serie", y="Value", hue="Currency type",ax=ax1,data=df_melted_nonkor)

g.legend(bbox_to_anchor=(-0.05, 1), loc=0, borderaxespad=0.)



#Using the dashes for korea data to stand out

g1=sns.lineplot(x="Time Serie", y="Value", color="coral",label="KOREA - WON/US$",ax=ax2,data=df_melted_kor)

g1.lines[0].set_linestyle("--")

g1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
kor_data = df_melted_kor.drop("Currency type",axis=1).rename(columns={'Time Serie': 'ds', 'Value': 'y'})
kor_data
#splitting 80% training and 20% testing



train_kor_data, test_kor_data = np.split(kor_data, [int(.8*len(df))])
from fbprophet import Prophet



prophet_basic=Prophet()

prophet_basic.fit(train_kor_data)

train_kor_future= prophet_basic.make_future_dataframe(periods=4*365)
train_kor_forecast=prophet_basic.predict(train_kor_future)

fig1 =prophet_basic.plot(train_kor_forecast)
from fbprophet.plot import add_changepoints_to_plot

fig = prophet_basic.plot(train_kor_forecast)

a = add_changepoints_to_plot(fig.gca(), prophet_basic, train_kor_forecast)
train_kor_forecast.head()
train_kor_yhat=train_kor_forecast['yhat']
train_kor_yhat_1460=train_kor_yhat[-1460:]

train_kor_yhat_1460
test_kor_data
merged_kor_data=test_kor_data.merge(train_kor_forecast,left_on='ds',right_on='ds')
merged_kor_data.head()
test_kor_data_2=merged_kor_data[['ds','y']].copy()

train_kor_yhat_2=merged_kor_data[['ds','yhat']].copy()
test_kor_data_2.isnull().sum()
test_kor_data_2['y']=test_kor_data_2['y'].fillna(test_kor_data_2['y'].median())
from sklearn.metrics import mean_squared_error

from math import sqrt

mse=mean_squared_error(test_kor_data_2["y"],train_kor_yhat_2["yhat"])

rmse=sqrt(mse)



print("RMSE :", rmse)
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))

plt.title("Korea Won currecy exchange rate predictions vs actual test")

predictions, = plt.plot(train_kor_yhat_2["ds"],train_kor_yhat_2["yhat"],label="Predictions")

test, = plt.plot(test_kor_data_2['ds'],test_kor_data_2['y'],label="Test")

plt.legend(loc='upper right')