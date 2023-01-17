# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb;

import matplotlib.pyplot as plt



from fbprophet import Prophet

from fbprophet.plot import add_changepoints_to_plot



from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



from sklearn.metrics import mean_squared_error

from math import sqrt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Cy_File=pd.read_csv("/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv")
Cy_File.head(10)
Cy_File=Cy_File.drop("Unnamed: 0",axis=1)

Cy_File.head(2)
Cy_File.describe()
Cy_File.info()
List=list(Cy_File)
Cy_File[List[1:]]=Cy_File[List[1:]].apply(pd.to_numeric ,errors='coerce')

#Coverting the elements into numeric format

Cy_File['Time Serie']=pd.to_datetime(Cy_File['Time Serie'])

#Then let's take care of Time series column data by converting it to date_time.

Cy_File.info()
#Creating a new Data Set

dataSet=Cy_File.melt(id_vars=["Time Serie"],var_name="Currency type",value_name="Value")

dataSet.head()
sb.set()

sb.set_style("white")

plt.figure(figsize=(20,10))

sb.lineplot(x="Time Serie", y="Value", hue="Currency type",palette='deep',data=dataSet)

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
data_Set_Kor=dataSet.loc[dataSet['Currency type'] == "KOREA - WON/US$"]

data_Set_rest=dataSet.loc[dataSet['Currency type'] != "KOREA - WON/US$"]
sb.set_style("white")

fig, ax1 = plt.subplots(figsize=(20,10))

ax2 = ax1.twinx()



#Using the dashes for not included korea data to stand out

Plot=sb.lineplot(x="Time Serie", y="Value", hue="Currency type",ax=ax1,data=data_Set_rest)

Plot.legend(bbox_to_anchor=(-0.05, 1), loc=0, borderaxespad=0.)



#Using the dashes for korea data to stand out

Plot2=sb.lineplot(x="Time Serie", y="Value", color="coral",label="KOREA - WON/US$",ax=ax2,data=data_Set_Kor)

Plot2.lines[0].set_linestyle("--")

Plot2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
data_Korea = data_Set_Kor.drop("Currency type",axis=1).rename(columns={'Time Serie': 'ds', 'Value': 'y'})

#Prophet requires you to have two columns, "ds" and "y" with the dates and values respectively.
data_Kor_Train,data_Kor_Test=np.split(data_Korea,[int(.8*len(Cy_File))])

#Let's Split data into 80% of training data and 20% of test data.
prophet_basic=Prophet()

prophet_basic.fit(data_Kor_Train)

future_train_kor= prophet_basic.make_future_dataframe(periods=5*365)
forecast_train_kor=prophet_basic.predict(future_train_kor)

Plot =prophet_basic.plot(forecast_train_kor)
fig = prophet_basic.plot(forecast_train_kor)

a = add_changepoints_to_plot(fig.gca(), prophet_basic, forecast_train_kor)
forecast_train_kor.head()

#Let's look at the predicted train dataframe.
#Now extracting the yhat values and saving it to a series.

train_kor_yhat=forecast_train_kor['yhat']

train_kor_yhat
#Look at our test dataset.

data_Kor_Test
merged_data_korea=data_Kor_Test.merge(forecast_train_kor,left_on='ds',right_on='ds')
korea_data_test2=merged_data_korea[['ds','y']].copy()

kor_yhat_train2=merged_data_korea[['ds','yhat']].copy()
korea_data_test2.isnull().sum()
korea_data_test2['y']=korea_data_test2['y'].fillna(korea_data_test2['y'].median())
mse=mean_squared_error(korea_data_test2["y"],kor_yhat_train2["yhat"])

rmse=sqrt(mse)



print("RMSE :", rmse)
#Let's plot both preditions and actual test data together.

plt.figure(figsize=(25,10))

plt.title("(Korea currecy exchange rate) predictions vs actual test")

predictions, = plt.plot(kor_yhat_train2["ds"],kor_yhat_train2["yhat"],label="Predictions")

test, = plt.plot(korea_data_test2['ds'],korea_data_test2['y'],label="Actual Test")

plt.legend(loc='lower right')