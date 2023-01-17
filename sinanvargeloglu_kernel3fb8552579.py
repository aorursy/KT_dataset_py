# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings

warnings.filterwarnings('ignore')

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb

import datetime as dt

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from datetime import timedelta

from sklearn.preprocessing import PolynomialFeatures 

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from statsmodels.tsa.api import Holt

from sklearn.metrics import mean_squared_error

covid=pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

covid.head()
covid.tail()
print("Size/Shape of the dataset", covid.shape)

print("Null values ", covid.isnull().sum())

print("Data types", covid.dtypes)
# Dropping the column

covid.drop(["SNo"],1,inplace=True)

covid.isnull().sum()

covid["ObservationDate"]=pd.to_datetime(covid["ObservationDate"])

covid["ObservationDate"]
grouped_country=covid.groupby(["Country/Region","ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

# Grouping different types of cases per date

datewise=covid.groupby(["ObservationDate"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
print("Basic information")

print("Total number of confirmed cases around the world", datewise["Confirmed"].iloc[-1])

print("Total number of recovered cases around the world", datewise["Recovered"].iloc[-1])

print("Total number of deaths cases around the world", datewise["Deaths"].iloc[-1])

print("Total number of active cases around the world",(datewise["Confirmed"].iloc[-1]-datewise["Recovered"].iloc[-1]-datewise["Deaths"].iloc[-1]))

print("Total number of closed cases around the world",(datewise["Recovered"].iloc[-1]+datewise["Deaths"].iloc[-1]))
plt.figure(figsize=(15,5))

sb.barplot(x=datewise.index.date,y=datewise["Confirmed"]-datewise["Recovered"]-datewise["Deaths"])

plt.title("Distributions Plot Active Cases")

plt.xticks(rotation=90)
datewise["WeekofYear"]=datewise.index.weekofyear

week_num=[]

weekwise_confirmed=[]

weekwise_recovered=[]

weekwise_deaths=[]

w=1

for i in  list(datewise["WeekofYear"].unique()):

  weekwise_confirmed.append(datewise[datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])

  weekwise_recovered.append(datewise[datewise["WeekofYear"]==i]["Recovered"].iloc[-1])

  weekwise_deaths.append(datewise[datewise["WeekofYear"]==i]["Deaths"].iloc[-1])

  week_num.append(w)

  w=w+1

plt.figure(figsize=(8,5))

plt.plot(week_num,weekwise_confirmed,linewidth=3)

plt.plot(week_num,weekwise_recovered,linewidth=3)

plt.plot(week_num,weekwise_deaths,linewidth=3)

plt.xlabel("Week Number")

plt.ylabel("Number of Cases")

plt.title("Weekly Progress of Different Types of Cases")
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(20,5))

sb.barplot(x=week_num,y=pd.Series(weekwise_confirmed).diff().fillna(0),ax=ax1)

sb.barplot(x=week_num,y=pd.Series(weekwise_deaths).diff().fillna(0),ax=ax2)

sb.barplot(x=week_num,y=pd.Series(weekwise_recovered).diff().fillna(0),ax=ax3)

ax1.set_xlabel("Week Number")

ax2.set_xlabel("Week Number")

ax1.set_ylabel("Number of Confirmed Cases")

ax2.set_ylabel("Number of Confirmed Cases")

ax1.set_title("Weekly Increase in Number of Confirmed Cases")

ax2.set_title("Weekly Increase in Number of Death Cases")

ax3.set_xlabel("Week Number")

ax3.set_ylabel("Number of Confirmed Cases")

ax3.set_title("Weekly Increase in Number of Recovered Cases")

plt.show()
print("Average increase in number of confirmed cases everyday:",np.round(datewise["Confirmed"].diff().fillna(0).mean()))

print("Average increase in number of recovered cases everyday:",np.round(datewise["Recovered"].diff().fillna(0).mean()))

print("Average increase in number of deaths cases everyday:",np.round(datewise["Deaths"].diff().fillna(0).mean()))



plt.figure(figsize=(15,6))

plt.plot(datewise["Confirmed"].diff().fillna(0),label="Daily Increase in Confirmed Cases",linewidth=3)

plt.plot(datewise["Recovered"].diff().fillna(0),label="Daily Increase in Recovered Cases",linewidth=3)

plt.plot(datewise["Deaths"].diff().fillna(0),label="Daily Increase in Death Cases",linewidth=3)

plt.xlabel("Timestamp")

plt.ylabel("Daily Increase")

plt.title("Daily Increase")

plt.legend()

plt.xticks(rotation=90)

plt.show()
countrywise=covid[covid["ObservationDate"]==covid["ObservationDate"].max()].groupby(["Country/Region"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"}).sort_values(["Confirmed"],ascending=False)

countrywise["Mortality"]=(countrywise["Deaths"]/countrywise["Recovered"])*100

countrywise["Recovery"]=(countrywise["Recovered"]/countrywise["Confirmed"])*100



fig,(ax1,ax2)=plt.subplots(1,2,figsize=(25,10))

top_15confirmed=countrywise.sort_values(["Confirmed"],ascending=False).head(10)

top_15deaths=countrywise.sort_values(["Deaths"],ascending=False).head(10)

sb.barplot(x=top_15confirmed["Confirmed"],y=top_15confirmed.index,ax=ax1)

ax1.set_title("Top 15 Countries per Number of Confirmed Cases")

sb.barplot(x=top_15deaths["Deaths"],y=top_15deaths.index,ax=ax2)

ax1.set_title("Top 15 Countries per Number of Death Cases")
covid_turkey=covid[covid['Country/Region']=="Turkey"]

covid_turkey["ObservationDate"]=pd.to_datetime(covid_turkey["ObservationDate"])

turkey_datewise=covid_turkey.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

turkey_datewise.head(10)
turkey_datewise["WeekofYear"]=turkey_datewise.index.weekofyear

turkey_datewise["Days Since"]=(turkey_datewise.index-turkey_datewise.index[0])

turkey_datewise["Days Since"]=turkey_datewise["Days Since"].dt.days

print("Number of Confirmed Cases",turkey_datewise["Confirmed"].iloc[-1])

print("Number of Recovered Cases",turkey_datewise["Recovered"].iloc[-1])

print("Number of Death Cases",turkey_datewise["Deaths"].iloc[-1])

print("Number of Active Cases",turkey_datewise["Confirmed"].iloc[-1]-turkey_datewise["Recovered"].iloc[-1]-turkey_datewise["Deaths"].iloc[-1])

print("Number of Closed Cases",turkey_datewise["Recovered"].iloc[-1]+turkey_datewise["Deaths"].iloc[-1])

print("Approximate Number of Confirmed Cases per day",round(turkey_datewise["Confirmed"].iloc[-1]/turkey_datewise.shape[0]))

print("Approximate Number of Recovered Cases per day",round(turkey_datewise["Recovered"].iloc[-1]/turkey_datewise.shape[0]))

print("Approximate Number of Death Cases per day",round(turkey_datewise["Deaths"].iloc[-1]/turkey_datewise.shape[0]))

print("Number of New Cofirmed Cases in last 24 hours are",turkey_datewise["Confirmed"].iloc[-1]-turkey_datewise["Confirmed"].iloc[-2])

print("Number of New Recoverd Cases in last 24 hours are",turkey_datewise["Recovered"].iloc[-1]-turkey_datewise["Recovered"].iloc[-2])

print("Number of New Death Cases in last 24 hours are",turkey_datewise["Deaths"].iloc[-1]-turkey_datewise["Deaths"].iloc[-2])
fig=px.bar(x=turkey_datewise.index,y=turkey_datewise["Confirmed"]-turkey_datewise["Recovered"]-turkey_datewise["Deaths"])

fig.update_layout(title="Distribution of Number of Active Cases",

                  xaxis_title="Date",yaxis_title="Number of Cases",)

fig.show()
fig=px.bar(x=turkey_datewise.index,y=turkey_datewise["Recovered"]+turkey_datewise["Deaths"])

fig.update_layout(title="Distribution of Number of Closed Cases",

                  xaxis_title="Date",yaxis_title="Number of Cases")

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=turkey_datewise.index, y=turkey_datewise["Confirmed"],

                    mode='lines+markers',

                    name='Confirmed Cases'))

fig.add_trace(go.Scatter(x=turkey_datewise.index, y=turkey_datewise["Recovered"],

                    mode='lines+markers',

                    name='Recovered Cases'))

fig.add_trace(go.Scatter(x=turkey_datewise.index, y=turkey_datewise["Deaths"],

                    mode='lines+markers',

                    name='Death Cases'))

fig.update_layout(title="Growth of different types of cases in Turkey",

                 xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
print('Mean Recovery Rate: ',((turkey_datewise["Recovered"]/turkey_datewise["Confirmed"])*100).mean())

print('Mean Mortality Rate: ',((turkey_datewise["Deaths"]/turkey_datewise["Confirmed"])*100).mean())

print('Median Recovery Rate: ',((turkey_datewise["Recovered"]/turkey_datewise["Confirmed"])*100).median())

print('Median Mortality Rate: ',((turkey_datewise["Deaths"]/turkey_datewise["Confirmed"])*100).median())

week_num_turkey=[]

turkey_weekwise_confirmed=[]

turkey_weekwise_recovered=[]

turkey_weekwise_deaths=[]

w=1

for i in list(turkey_datewise["WeekofYear"].unique()):

    turkey_weekwise_confirmed.append(turkey_datewise[turkey_datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])

    turkey_weekwise_recovered.append(turkey_datewise[turkey_datewise["WeekofYear"]==i]["Recovered"].iloc[-1])

    turkey_weekwise_deaths.append(turkey_datewise[turkey_datewise["WeekofYear"]==i]["Deaths"].iloc[-1])

    week_num_turkey.append(w)

    w=w+1
print("Average weekly increase in number of Confirmed Cases",round(pd.Series(turkey_weekwise_confirmed).diff().fillna(0).mean()))

print("Average weekly increase in number of Recovered Cases",round(pd.Series(turkey_weekwise_recovered).diff().fillna(0).mean()))

print("Average weekly increase in number of Death Cases",round(pd.Series(turkey_weekwise_deaths).diff().fillna(0).mean()))



fig = make_subplots(rows=2, cols=3)

fig.add_trace(

    go.Bar(x=week_num_turkey, y=pd.Series(turkey_weekwise_confirmed).diff().fillna(0),

          name="Weekly rise in number of Confirmed Cases"),

    row=1, col=2

)



fig.add_trace(

    go.Bar(x=week_num_turkey, y=pd.Series(turkey_weekwise_recovered).diff().fillna(0),

          name="Weekly rise in number of Confirmed Cases"),

    row=2, col=1

)



fig.add_trace(

    go.Bar(x=week_num_turkey, y=pd.Series(turkey_weekwise_deaths).diff().fillna(0),

          name="Weekly rise in number of Death Cases"),

    row=2, col=3

)

fig.update_layout(title="Turkey's Weekly increas in Number of Confirmed, Revored and Death Cases",

    font=dict(

        size=10,

    )

)

fig.update_layout(width=1000,legend=dict(x=0,y=-0.5,traceorder="normal"))

fig.update_xaxes(title_text="Week Based Date", row=1, col=2)

fig.update_yaxes(title_text="Number of Cases", row=1, col=2)

fig.update_xaxes(title_text="Week Based Date", row=2, col=1)

fig.update_yaxes(title_text="Number of Cases", row=2, col=1)

fig.update_xaxes(title_text="Week Based Date", row=2, col=3)

fig.update_yaxes(title_text="Number of Cases", row=2, col=3)

fig.show()
train_ml=turkey_datewise.iloc[:int(turkey_datewise.shape[0]*0.95)]

valid_ml=turkey_datewise.iloc[int(turkey_datewise.shape[0]*0.95):]

model_scores=[]
plt.figure(figsize=(11,6))

fig=go.Figure()

fig.add_trace(go.Scatter(x=turkey_datewise.index, y=turkey_datewise["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases",))

fig.update_layout(title="Confirmed Cases of Change in Turkey",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
linef = PolynomialFeatures(degree = 5)

train_linef=linef.fit_transform(np.array(train_ml["Days Since"]).reshape(-1,1))

valid_linef=linef.fit_transform(np.array(valid_ml["Days Since"]).reshape(-1,1))

y=train_ml["Confirmed"]
linreg=LinearRegression(normalize=True)

linreg.fit(train_linef,y)
prediction_lin=linreg.predict(valid_linef)

rmse_poly=np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_lin))

model_scores.append(rmse_poly)

print("Root Mean Squared Error for Polynomial Regression: ",rmse_poly)
comp_data=linef.fit_transform(np.array(turkey_datewise["Days Since"]).reshape(-1,1))

plt.figure(figsize=(11,6))

predictions_lin=linreg.predict(comp_data)

fig=go.Figure()

fig.add_trace(go.Scatter(x=turkey_datewise.index, y=turkey_datewise["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=turkey_datewise.index, y=predictions_lin,

                    mode='lines',name="Polynomial Regression",

                    line=dict(color='black', dash='dot')))

fig.update_layout(title="Confirmed Cases Polynomial Regression Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
new_date=[]

new_prediction_poly=[]

for i in range(1,30):

    new_date.append(turkey_datewise.index[-1]+timedelta(days=i))

    new_date_poly=linef.fit_transform(np.array(turkey_datewise["Days Since"].max()+i).reshape(-1,1))

    new_prediction_poly.append(linreg.predict(new_date_poly)[0])
model_predictions=pd.DataFrame(zip(new_date,new_prediction_poly),columns=["Date For Turkey:","Polynomial Regression Prediction"])

model_predictions.head(5)
train_ml=turkey_datewise.iloc[:int(turkey_datewise.shape[0]*0.95)]

valid_ml=turkey_datewise.iloc[int(turkey_datewise.shape[0]*0.95):]

svm=SVR(kernel='rbf', C=1e6, gamma='scale')

svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),train_ml["Confirmed"])
prediction_svm=svm.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))

rmse_svm=np.sqrt(mean_squared_error(prediction_svm,valid_ml["Confirmed"]))

model_scores.append(rmse_svm)

print("Root Mean Square Error for SVR Model: ",rmse_svm)
plt.figure(figsize=(11,6))

predictions=svm.predict(np.array(turkey_datewise["Days Since"]).reshape(-1,1))

fig=go.Figure()

fig.add_trace(go.Scatter(x=turkey_datewise.index, y=turkey_datewise["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=turkey_datewise.index, y=predictions,

                    mode='lines',name="Support Vector Machine Best fit Kernel",

                    line=dict(color='black', dash='dot')))

fig.update_layout(title="Confirmed Cases Support Vectore Machine Regressor Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
new_date=[]

new_prediction_svm=[]

for i in range(1,30):

    new_date.append(turkey_datewise.index[-1]+timedelta(days=i))

    new_prediction_svm.append(svm.predict(np.array(turkey_datewise["Days Since"].max()+i).reshape(-1,1))[0])

model_predictions["SVM Prediction"]=new_prediction_svm

model_predictions.head(5)
model_train=turkey_datewise.iloc[:int(turkey_datewise.shape[0]*0.95)]

valid=turkey_datewise.iloc[int(turkey_datewise.shape[0]*0.95):]

y_pred=valid.copy()
holt=Holt(np.asarray(model_train["Confirmed"])).fit(smoothing_level=0.7, smoothing_slope=1.2)
y_pred["Holt"]=holt.forecast(len(valid))

rmse_holt_linear=np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt"]))

model_scores.append(rmse_holt_linear)

print("Root Mean Square Error Holt's Linear Model: ",rmse_holt_linear)
fig=go.Figure()

fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=valid.index, y=y_pred["Holt"],

                    mode='lines+markers',name="Prediction of Confirmed Cases",))

fig.update_layout(title="Confirmed Cases Holt's Linear Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
holt_new_prediction=[]

for i in range(1,30):

    holt_new_prediction.append(holt.forecast((len(valid)+i))[-1])



model_predictions["Holt's Linear Model Prediction"]=holt_new_prediction

model_predictions.head(5)
