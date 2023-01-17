## Import libs.
import numpy as np
import pandas as pd 
import scipy as py
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import MinMaxScaler
from fbprophet import Prophet
from scipy.interpolate import *
from numpy import linspace,exp
import warnings

## Filter warnings
warnings.filterwarnings("ignore")
## Set display options
pd.set_option('display.max_columns', None)

## Read the data
df = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")
## Drop "Province/State"
df = df.drop(columns = ["Province/State"])

## Rename "Country/Region"
df = df.rename(columns = {"Country/Region":"Country"})

## Convert the data type of "Date"
df.Date = df.Date.map(lambda x: x + "20") # Add "20" to make ...2020
df["Date"]=pd.to_datetime(df["Date"],format="%m/%d/%Y") # Convert str to datetime

## Sum all the values of Countries (without state)
df = pd.DataFrame(df.groupby(["Country","Date"]).sum()).reset_index()
## Get the ratio of died to the confirmed by country
df["Fatalite"] = df.Deaths / df.Confirmed

## Get the ratio of recovered to the confirmed by country
df["Survival"] = df.Recovered / df.Confirmed
## China case
summary = pd.DataFrame(df[df["Country"]=="China"].groupby(["Date"])[["Confirmed","Deaths","Recovered","Fatalite","Survival"]].sum()).reset_index()

fig = go.Figure()
fig.add_trace(go.Scatter(x=summary.Date, y=summary.Confirmed, name="Confirmed"))
fig.add_trace(go.Scatter(x=summary.Date, y=summary.Deaths, name="Deaths"))
fig.add_trace(go.Scatter(x=summary.Date, y=summary.Recovered, name="Recovered"))
fig.add_trace(go.Scatter(x=summary.Date, y=np.log(summary.Confirmed), name="Log Confirmed"))
fig.add_trace(go.Scatter(x=summary.Date, y=np.log(summary.Deaths), name="Log Deaths"))
fig.add_trace(go.Scatter(x=summary.Date, y=np.log(summary.Recovered), name="Log Recovered"))
fig.show()
## China Case - Fatalite x Survival
fig = go.Figure()
fig.add_trace(go.Scatter(x=summary.Date, y=summary.Fatalite, name="Fatalite"))
fig.add_trace(go.Scatter(x=summary.Date, y=summary.Survival, name="Survival"))
fig.show()
## Calculate risk level based on current data


## Get only last values
summary = df.groupby("Country")[["Survival","Fatalite","Confirmed"]].apply(lambda x: x.tail(1)).reset_index()

## Calculate duration
duration = pd.DataFrame(df[df.Confirmed != 0].groupby("Country")[["Confirmed"]].count()).reset_index().sort_values("Confirmed")
duration = duration.rename(columns={"Confirmed":"t"})
summary = pd.merge(summary, duration, on = "Country", how = "left") # Merge data

## Calculate spreading rate
summary["Spreading"] = np.log(summary.Confirmed + 1)/summary.t

## Calculate success rate 
summary["Success"] = (summary.Survival + 1)/(summary.Fatalite+1)

## Calculate risk level
summary["Risk_Level"] = summary.Spreading ** 1/summary.Success
summary = summary.sort_values(["Risk_Level"],ascending=False)
summary = summary.drop(columns = ["level_1"])
summary = summary.fillna(0)
## Top 10 most risky countries
summary.head(10)
## Top 10 least risk countries
summary.tail(10)
## Highly infected countries
summary[summary.Country.isin(["US","Italy","China","Spain","Germany","South Korea","Turkey"])]
## Bubble chart
fig = px.scatter(summary.sort_values("Confirmed",ascending=False), 
                 x="Survival", y="Fatalite", size="Confirmed", hover_name="Country",
                 color="Country",log_x=True,log_y=True,size_max=50)
fig.show()
## Clustering based on Survival, Fatalite, Confirmed (Bubble chart features)
features = ["Survival","Fatalite","Confirmed"]
df_array = MinMaxScaler().fit_transform(summary[features])
labels = summary.Country.tolist()

import numpy as np
fig = ff.create_dendrogram(df_array,orientation="left",labels=labels, color_threshold=0.8,
                          linkagefun=lambda x: linkage(df_array, 'ward', metric='euclidean'))
fig.update_layout(width=1000, height=600)
fig.show()
## Clustering based on current Spreading, Success, Risk_Level
features = ["Spreading","Success","Risk_Level"]
df_array = MinMaxScaler().fit_transform(summary[features])
labels = summary.Country.tolist()

import numpy as np
fig = ff.create_dendrogram(df_array,orientation="left",labels=labels, color_threshold=0.45,
                          linkagefun=lambda x: linkage(df_array, 'ward', metric='euclidean'))
fig.update_layout(width=1000, height=600)
fig.show()
## Curve fitting for China
E = np.inf
# Confirmed rate of China has polynomial curve. 
# It seems (based on sigmoid curve of confirmed rate) that confirmation rate is third degree polynomial.
# It is expected that higher value better fit to the curve.
# The range is defined from 1 to 4 to find the best value without overfitting.
for i in range(1,4):
    
    # Select only China
    x = df[df.Country == "China"][["Date","Confirmed"]].reset_index(drop=True)
    
    # Fit on n degree polynomial curve
    pl = np.polyfit(x.index, x.Confirmed,i)
    
    # Decide estimated length
    n=len(x)
    
    # Estimation
    yhat = np.polyval(pl,range(n))
    
    # Mean squared error of estimation
    mse = ((x.Confirmed - yhat)**(0.5)).mean()
    
    if mse < E:
        E = mse
        best_degree = i
        
print(best_degree)
## Prediction for confirmation rate of Turkey 

# Gisanddata for confirmation rate of Turkey
# https://gisanddata.maps.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6
x = pd.DataFrame()
x["ds"] = ["11.03.2020","12.03.2020","13.03.2020","14.03.2020","15.03.2020","16.03.2020","17.03.2020",
          "18.03.2020","19.03.2020","20.03.2020","21.03.2020","22.03.2020","23.03.2020","24.03.2020",
          "25.03.2020","26.03.2020","27.03.2020","28.03.2020","29.03.2020","30.03.2020","31.03.2020",
          "01.04.2020"]
"""x["y"] = pd.DataFrame([1,1,5,5,6,18,47,98,192,359,670,1200,1500,1900,
          2400,3629,5698,7402,9217,10827,13531,15679]).rolling(window=2).mean()"""
x["y"] = [1,1,5,5,6,18,47,98,192,359,670,1200,1500,1900,
          2400,3629,5698,7402,9217,10827,13531,15679]
x["ds"] = pd.to_datetime(x["ds"],format='%d.%m.%Y')
x = x.fillna(0)

# Fit polynomial curve with best_degree
pl = np.polyfit(x.index, x.y,best_degree)

# Prediction range (lenght of data + 3 day)
n=len(x)+3

# Create dataframe for easier observe the results 
yhat = pd.DataFrame()
yhat["Pred"] = np.polyval(pl,range(n))
yhat["Date"] = pd.date_range(x.ds[0], periods=n).tolist()

# Graph
fig = go.Figure()
fig.add_trace(go.Scatter(x = x.ds, y=x.y, name="Yayılma"))
fig.add_trace(go.Scatter(x = yhat.Date, y=yhat.Pred, name="Tahmin"))
fig.show()

print("Estimated Values: ")
print(yhat)
## Make prophet prediction that has best fitted capacity value

x = pd.DataFrame()
x["ds"] = ["11.03.2020","12.03.2020","13.03.2020","14.03.2020","15.03.2020","16.03.2020","17.03.2020",
          "18.03.2020","19.03.2020","20.03.2020","21.03.2020","22.03.2020","23.03.2020","24.03.2020",
          "25.03.2020","26.03.2020","27.03.2020","28.03.2020","29.03.2020","30.03.2020","31.03.2020",
          "01.04.2020"]
x["y"] = pd.DataFrame([1,1,5,5,6,18,47,98,192,359,670,1200,1500,1900,2400,3629,5698,7402,9217,10827,
                       13531,15679])
x["ds"] = pd.to_datetime(x1["ds"],format='%d.%m.%Y')

# Define best fitted capacity value
x["cap"] = 35000
# Model fit
model_fit = Prophet(growth="logistic", 
                    yearly_seasonality = False, 
                    weekly_seasonality = False, 
                    daily_seasonality = False
                   ).fit(x)
# Define time period
future=model_fit.make_future_dataframe(periods=3)
# Add capacity
future["cap"] = 35000
# Make forecast for defined periods
forecast=model_fit.predict(future)

# Graph
fig = model_fit.plot(forecast)
plt.show()

# Show the predicted values
print("Predicted Values")
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)