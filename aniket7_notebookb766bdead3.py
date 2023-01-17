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
import numpy as np 
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from fbprophet import Prophet
import seaborn as sns
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
df.head()
df.columns
df.hotel.value_counts()
hoteltype=df.groupby("hotel").is_canceled.count().reset_index()
hoteltype.columns=["hotel","count"]
sns.set(style="whitegrid")
ax = sns.barplot(x="hotel", y="count", data=hoteltype)
cancelp=df.groupby(["hotel","is_canceled"]).lead_time.count().reset_index()
cancelp.columns=["hotel","is_canceled","count"]
ax = sns.barplot(x="hotel", y="count", hue="is_canceled", data=cancelp)
df["reservation_status_date"]=pd.to_datetime(df["reservation_status_date"])
is_cancelled_plot=df.groupby("reservation_status_date").is_canceled.sum().reset_index().sort_values(by=["reservation_status_date"])
is_cancelled_plot.head()
is_cancelled_plot.dtypes
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(ax=ax,x="reservation_status_date", y="is_canceled", data=is_cancelled_plot)
is_cancelled_plot.tail()
days = pd.date_range("2015-01-01", "2017-09-14", freq='D')
is_cancelled_fill=pd.DataFrame({"reservation_status_date":days})
is_cancelled_fill=pd.merge(is_cancelled_plot[2:],is_cancelled_fill,on="reservation_status_date", how="outer")
is_cancelled_fill=is_cancelled_fill.fillna(0)
is_cancelled_fill.head()
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(ax=ax,x="reservation_status_date", y="is_canceled", data=is_cancelled_fill)
AllindexOutlier=[]    
df_table = is_cancelled_fill["is_canceled"].copy()
Q1 = df_table.quantile(0.25)
Q3 = df_table.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1- 1.5*IQR
upper_bound = Q3 + 1.5*IQR
print("lower bound is " + str(lower_bound))
print("upper bound is " + str(upper_bound))
print(Q1)
print(Q3)
outliers_vector = (df_table < (lower_bound)) | (df_table > (upper_bound) )
outliers_vector
outliers = df_table[outliers_vector]
listOut=outliers.index.to_list()
for t in listOut:
    AllindexOutlier.append(t)
AllindexOutlier[0:15]
for i in AllindexOutlier:
    #I filled it with its average.
    is_cancelled_fill.loc[i,"is_canceled"]=is_cancelled_fill.is_canceled.mean()
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(ax=ax,x="reservation_status_date", y="is_canceled", data=is_cancelled_fill)
def test_train_split(data, test_split,datecol):
    test_train_ind = (data[datecol][data[datecol] < test_split].index.values,
                      data[datecol][data[datecol] >= test_split].index.values)
    data_train = data.iloc[test_train_ind[0],].reset_index(drop=True)
    data_test = data.iloc[test_train_ind[1],].reset_index(drop=True)
    return data_train, data_test
is_cancelled_fill.columns=["ds","y"]
data_train,data_test=test_train_split(is_cancelled_fill, "2017-01-01","ds")
data_test.head()
m = Prophet()
m.fit(data_train)
future = m.make_future_dataframe(periods=len(data_test))
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
data_test=pd.merge(data_test,forecast,on="ds",how="inner")
plt.figure(figsize=(20,10))
lines = plt.plot(data_test.ds, data_test.y, data_test.ds, data_test.yhat)
plt.setp(lines[0], linewidth=2)
plt.setp(lines[1], linewidth=2)

plt.legend(('y', 'yhat'),
           loc='upper right')
plt.title('Prediction')
plt.show()
