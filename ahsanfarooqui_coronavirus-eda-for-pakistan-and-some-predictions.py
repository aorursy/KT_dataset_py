import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")
data.head()

data.rename(columns={"Country/Region":"Country"},inplace=True)
confirmedcasessum  = data.groupby(["Country"])["ConfirmedCases"].sum().reset_index()

zeroconfirmcases = confirmedcasessum[confirmedcasessum.ConfirmedCases==0].Country.tolist()
for i in data["Country"].unique():

    if i not in (zeroconfirmcases):

        c_data = data[data["Country"]==i]

        #print(i,"%",c_data.iloc[next((x for x,y in enumerate(c_data.ConfirmedCases) if y),None)].Date)

        data.loc[data["Country"]==i,"FirstCaseDate"]=c_data.iloc[next((x for x,y in enumerate(c_data.ConfirmedCases) if y),None)].Date
for i in range(len(data)):

    

    try:

        current_date = data.loc[i,"Date"]

        current_date = datetime.datetime.strptime(current_date,"%Y-%m-%d")

        firstcase_date = data.loc[i,"FirstCaseDate"]

        firstcase_date = datetime.datetime.strptime(firstcase_date,"%Y-%m-%d")

        days = current_date-firstcase_date

        #print(days.days)

        data.loc[i,"daysSinceFirstCase"] = days.days

    except:

        continue
data = data[data.daysSinceFirstCase>=0]
confirmedcasessum.sort_values(by="ConfirmedCases",ascending=False,inplace=True)

top5countries = confirmedcasessum.head(5)["Country"].tolist()

top5countries.append("Pakistan")
top5countries.pop(top5countries.index("China"))
plt.figure(figsize=(12,8))

plt.title("Number of Confirmed Cases \n Since first day")

h = sns.lineplot(x="daysSinceFirstCase",y="ConfirmedCases",data=data[data.Country.isin(top5countries)],hue="Country")

h.set(yscale="log")
plt.figure(figsize=(12,8))

plt.title("Number of Fatalities \n Since first day")

h = sns.lineplot(x="daysSinceFirstCase",y="Fatalities",data=data[data.Country.isin(top5countries)],hue="Country")

h.set(yscale="log")
data.columns
pakdata =data[data.Country=="Pakistan"]
from scipy.optimize import curve_fit

def func(x, a, b, c):

    #return a * np.log(b*x) + c

    return a * np.exp(b * x) + c

popt, pcov = curve_fit(func, xdata=pakdata.daysSinceFirstCase, ydata=pakdata.ConfirmedCases)
plt.plot(pakdata.daysSinceFirstCase, func(pakdata.daysSinceFirstCase, *popt), 'r-',

         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.plot(pakdata.daysSinceFirstCase,pakdata.ConfirmedCases)

plt.title("coronavirus Pakistan and best fit line")

plt.plot()
from statsmodels.tsa.arima_model import ARIMA

pakdata_arima = pakdata[["Date","ConfirmedCases"]]
pakdata_arima.dtypes
pakdata_arima["Date"] = pd.to_datetime(pakdata_arima["Date"])
pakdata_arima.set_index("Date",inplace=True)
model = ARIMA(pakdata_arima, order=(1, 2, 2))
model_fit = model.fit(disp=False)

yhat = model_fit.predict(len(pakdata_arima), len(pakdata_arima)+5, typ='levels')

yhat
plt.plot(pakdata_arima.index,pakdata_arima.ConfirmedCases)

plt.plot(yhat)

plt.xticks(rotation=90)
pakdata