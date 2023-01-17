import numpy as np 

import pandas as pd 

import os

import plotly.express as px

import datetime

import matplotlib.pyplot as plt

import seaborn as sb

from statsmodels.tsa.arima_model import ARIMA

from sklearn.feature_selection import RFE

from statsmodels.graphics.tsaplots import plot_acf

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

import subprocess

import sys

def install(package):

    subprocess.check_call([sys.executable, "-m", "pip","install",package])

install("pmdarima")

import pmdarima as pm

import warnings

warnings.filterwarnings("ignore")
dataInicial = pd.read_csv("../input/cases2/data_comexstat.csv",encoding='latin-1') 

train = pd.read_excel("../input/cases2/covariates.xlsx")
dataInicial
print( "Labels in Type variable:",dataInicial["type"].unique())

print( "Labels in Product variable:",dataInicial["product"].unique())
dataInicial['month'] = pd.DatetimeIndex(dataInicial['date']).month

dataInicial['year'] = pd.DatetimeIndex(dataInicial['date']).year
products = ['soybean_meal', 'soybean_oil', 'soybeans']

for products_i in products:

    df = dataInicial.loc[(dataInicial["product"]== products_i) & (dataInicial["type"]=='Export')].groupby(["year","month"]).sum().reset_index()

    df["date"] = [datetime.datetime(year=df["year"][index],month=df["month"][index],day=1) for index in df.index] 

    fig = px.line(df, x="date", y="tons",title="Total monthly exports from Brazil (all states and to everywhere) of ‘soybeans’, ‘soybean oil’ and ‘soybean meal’ " )

    fig.show()
products = ['soybean_meal', 'soybean_oil', 'soybeans']

for products_i in products:

    df = dataInicial.loc[(dataInicial["product"]== products_i) & (dataInicial["type"]=='Export')].groupby(["year"]).sum().reset_index()

    fig = px.line(df, x="year", y="tons",title="Total annual exports from Brazil (all states and to everywhere) of " + products_i)

    fig.show()
year_to_compare = max(dataInicial.year)-5
df = dataInicial.loc[(dataInicial.year>year_to_compare)&(dataInicial.type=="Export"),:].groupby(["year","product"]).usd.mean().reset_index().sort_values("usd",ascending=False)

fig = px.treemap(df, path=["year",'product'], values='usd',color="usd",

                  title= "Amount in USD of products exported by Brazil in the last 5 years")

df = df.groupby("product").usd.sum().reset_index().sort_values("usd",ascending=False)

fig1 = px.bar(df, x='product', y='usd',color="usd",

                  title= "Total amount in USD of products exported by Brazil in the last 5 years")

fig.show()

fig1.update_layout(

    showlegend=False,

    annotations=[

        dict(

            x="corn",

            y= 50000000,

            xref="x",

            yref="y",

            text="The first three represent the 80.2166% of total amount generated ",

            showarrow=False,

            arrowhead=7,

            ax=0,

            ay=-40

        )

    ]

)

fig1.show()
year_to_compare = max(dataInicial.year)-5

print(" Main routes through which Brazil have been exporting ‘corn’ in the last 5 years ")

dataInicial.loc[(dataInicial["product"]=="corn")&(dataInicial.year>year_to_compare)&(dataInicial.type=="Export")].groupby("route").size().to_frame(name="frequenty").sort_values("frequenty",ascending=False)
df = dataInicial.loc[(dataInicial.year>year_to_compare)&(dataInicial.type=="Export")].groupby(["product","route"]).size().to_frame(name="frequenty").reset_index()

fig = px.treemap(df, path=['product','route'], values='frequenty',color="frequenty",

             title= " Main routes through which Brazil have been exporting in the last 5 years ")

fig.show()
year_to_compare = max(dataInicial.year)-3

df = dataInicial.loc[(dataInicial.year>year_to_compare)&((dataInicial["product"]=="corn")|(dataInicial["product"]=="sugar"))].groupby(["year","country"]).usd.sum().sort_values(ascending=False).reset_index()

df2017 = df.loc[df.year==2017].sort_values("usd",ascending=False).iloc[0:5,:]

df2018 = df.loc[df.year==2018].sort_values("usd",ascending=False).iloc[0:5,:]

df2019 = df.loc[df.year==2019].sort_values("usd",ascending=False).iloc[0:5,:]

df = df2017.append(df2018).append(df2019).reset_index().drop(columns="index")

fig = px.treemap(df, path=['year','country'], values='usd',

             title= "Top 5 countries that have been the most important trade partners (total trade) for Brazil in terms of ‘corn’ and ‘sugar’ in the last 3 years")

fig.show()
y_labels = dataInicial.groupby(["year","product"]).tons.sum().unstack(level=1, fill_value=0).reset_index().set_index("year").drop(columns=["soybean_oil","sugar","wheat"])

train = train.set_index("year")
Corn_tsPlot = px.line(y_labels.reset_index(), x="year", y="corn",

              title="Total annual exports from Brazil (all states and to everywhere) of corn ")

SoybeanMeal_tsPlot = px.line(y_labels.reset_index(), x="year", y="soybean_meal",

              title="Total annual exports from Brazil (all states and to everywhere) of soybean meal ")

Soybeans_tsPlot = px.line(y_labels.reset_index(), x="year", y="soybeans",

              title="Total annual exports from Brazil (all states and to everywhere) of soybeans ")

Corn_tsPlot.show()

SoybeanMeal_tsPlot.show()

Soybeans_tsPlot.show()
df = train.iloc[:,0:3].unstack(level=1, fill_value=0).to_frame(name="price").reset_index()

fig = px.line(df, x="year", y="price",color="level_0",

              title="Annual price of soybeans, soybean meal and corn")

fig.show()
df = train.iloc[:,3:-3].unstack(level=1, fill_value=0).to_frame(name="GDP").reset_index()

fig = px.line(df, x="year", y="GDP",color="level_0",

              title="Annual Gross Domestic Product of some countries")

fig.show()
second_scheme = train.join(y_labels).loc[train.index>1996,:].copy()

forecastData = second_scheme.loc[second_scheme.index>=2020,:].drop(columns=["corn","soybean_meal","soybeans"])

trainData = second_scheme.loc[second_scheme.index<2020,:]
cutoff = 3

trainSet = trainData.iloc[:-cutoff,:].copy()

validationSet = trainData.iloc[-cutoff:,:].copy()
trainSet.shape
f,ax=plt.subplots(2,3,figsize=(18,8))

plot_acf(trainSet["corn"],ax=ax[0,0])

ax[0,0].set_title('ACF for corn series')

ax[0,0].set_ylabel('')

plot_acf(trainSet["soybeans"],ax=ax[0,1])

ax[0,1].set_title('ACF for soybeans series ')

ax[0,1].set_ylabel('')

plot_acf(trainSet["soybean_meal"],ax=ax[0,2])

ax[0,2].set_title('ACF for soybean_meal series ')

ax[0,2].set_ylabel('')



plot_acf(trainSet["corn"].diff(),missing="drop",ax=ax[1,0])

ax[1,0].set_title('ACF for corn series after diff')

ax[1,0].set_ylabel('')

plot_acf(trainSet["soybeans"].diff(),missing="drop",ax=ax[1,1])

ax[1,1].set_title('ACF for soybeans series after diff ')

ax[1,1].set_ylabel('')

plot_acf(trainSet["soybean_meal"].diff(),missing="drop",ax=ax[1,2])

ax[1,2].set_title('ACF for soybean_meal series after diff ')

ax[1,2].set_ylabel('')

plt.show()

modelArimaCorn = pm.auto_arima(trainSet["corn"])

print("AIC corn series:",modelArimaCorn.aic(),"with order",modelArimaCorn.order)

modelArimaSoybeans = pm.auto_arima(trainSet["soybeans"])

print("AIC soybeans series:",modelArimaSoybeans.aic(),"with order",modelArimaSoybeans.order)

modelArimaSoybean_meal = pm.auto_arima(trainSet["soybean_meal"])

print("AIC soybean_meal series:",modelArimaSoybean_meal.aic(),"with order",modelArimaSoybean_meal.order)
X_train= trainSet.iloc[:,:-3]

rfe = RFE(RandomForestRegressor(random_state=1997))            

rfe = rfe.fit(X_train, trainSet["corn"])

covariatesCorn = X_train.columns[rfe.support_]           

rfe = rfe.fit(X_train, trainSet["soybeans"])

covariatesSoybeans = X_train.columns[rfe.support_]               

rfe = rfe.fit(X_train, trainSet["soybean_meal"])

covariatesSoybean_meal = X_train.columns[rfe.support_]  
modelArimaxCorn = pm.auto_arima(trainSet["corn"], trainSet.loc[:,covariatesCorn])

print("AIC corn series:",modelArimaxCorn.aic(),"with order",modelArimaxCorn.order)

modelArimaxSoybeans = pm.auto_arima(trainSet["soybeans"],  trainSet.loc[:,covariatesSoybeans])

print("AIC soybeans series:",modelArimaxSoybeans.aic(),"with order",modelArimaxSoybeans.order)

modelArimaxSoybean_meal = pm.auto_arima(trainSet["soybean_meal"], trainSet.loc[:,covariatesSoybean_meal])

print("AIC soybean_meal series:",modelArimaxSoybean_meal.aic(),"with order",modelArimaxSoybean_meal.order)
train = train.reset_index()

train["year"] = train.year+11

### Join and split into train set and forcasting set

third_scheme = train.set_index("year").join(y_labels).copy()

third_scheme = third_scheme.loc[third_scheme.index>1996,:]

forecastDataThirdScheme = third_scheme.loc[third_scheme.index>=2020,:].drop(columns=["corn","soybean_meal","soybeans"])

forecastDataThirdScheme = forecastDataThirdScheme.loc[forecastDataThirdScheme.index<2031,:]

trainDataThirdScheme = third_scheme.loc[third_scheme.index<2020,:]

### Join and split into train set and validation set

cutoff = 3

trainSetThirdScheme = trainDataThirdScheme.iloc[:-cutoff,:].copy()

validationSetThirdScheme = trainDataThirdScheme.iloc[-cutoff:,:].copy()
X_train= trainSetThirdScheme.iloc[:,:-3]

rfe = RFE(RandomForestRegressor(random_state=1997))            

rfe = rfe.fit(X_train, trainSetThirdScheme["corn"])

covariatesCorn1 = X_train.columns[rfe.support_]           

rfe = rfe.fit(X_train, trainSetThirdScheme["soybeans"])

covariatesSoybeans1 = X_train.columns[rfe.support_]               

rfe = rfe.fit(X_train, trainSetThirdScheme["soybean_meal"])

covariatesSoybean_meal1 = X_train.columns[rfe.support_]  
modelArimaxCornThirdsSheme = pm.auto_arima(trainSetThirdScheme["corn"], trainSetThirdScheme.loc[:,covariatesCorn1])

print("AIC corn series:",modelArimaxCorn.aic(),"with order",modelArimaxCorn.order)

modelArimaxSoybeansThirdsSheme = pm.auto_arima(trainSetThirdScheme["soybeans"],  trainSetThirdScheme.loc[:,covariatesSoybeans1])

print("AIC soybeans series:",modelArimaxSoybeans.aic(),"with order",modelArimaxSoybeans.order)

modelArimaxSoybean_mealThirdsSheme = pm.auto_arima(trainSetThirdScheme["soybean_meal"], trainSetThirdScheme.loc[:,covariatesSoybean_meal1])

print("AIC soybean_meal series:",modelArimaxSoybean_meal.aic(),"with order",modelArimaxSoybean_meal.order)
def MSLE_evaluation_arima(model,label): 

    predicted = model.predict(cutoff)        

    real = validationSet[label]

    msle = np.log(mean_squared_error(real, predicted))

    return msle.round(3)





def MSLE_evaluation_arimax(model,label_x,label): 

    predicted = model.predict(cutoff,validationSet.loc[:,label_x])

    real = validationSet[label]

    msle = np.log(mean_squared_error(real, predicted))

    return msle.round(3)



def MSLE_evaluation_arimax_1(model,label_x,label): 

    predicted = model.predict(cutoff,validationSetThirdScheme.loc[:,label_x])

    real = validationSet[label]

    msle = np.log(mean_squared_error(real, predicted))

    return msle.round(3)



print("MSLE for CORN series without covariates: ",MSLE_evaluation_arima(modelArimaCorn,"corn")

      ,"with covariates:",MSLE_evaluation_arimax(modelArimaxCorn,covariatesCorn,"corn")

     ,"with covariates third scheme:",MSLE_evaluation_arimax_1(modelArimaxCornThirdsSheme,covariatesCorn1,"corn"))



print("MSLE for Soybeans series without covariates: ",MSLE_evaluation_arima(modelArimaSoybeans,"soybeans")

      ,"with covariates:",MSLE_evaluation_arimax(modelArimaxSoybeans,covariatesSoybeans,"soybeans")

      ,"with covariates third scheme:",MSLE_evaluation_arimax_1(modelArimaxSoybeansThirdsSheme,covariatesSoybeans1,"soybeans"))



print("MSLE for Soybean_meal series without covariates: ",MSLE_evaluation_arima(modelArimaSoybean_meal,"soybean_meal")

      ,"with covariates:",MSLE_evaluation_arimax(modelArimaxSoybean_meal,covariatesSoybean_meal,"soybean_meal")

      ,"with covariates third scheme:",MSLE_evaluation_arimax_1(modelArimaxSoybean_mealThirdsSheme,covariatesSoybean_meal1,"soybean_meal"))
modelArimaCorn = ARIMA(endog= trainDataThirdScheme["corn"],exog = trainDataThirdScheme.loc[:,covariatesCorn1],order=(0, 0, 0)).fit(disp=0)

modelArimaSoybeans = ARIMA(endog= trainDataThirdScheme["soybeans"],exog = trainDataThirdScheme.loc[:,covariatesSoybeans1],order=(0,0,1)).fit(disp=0)

modelArimaSoybean_meal = ARIMA(endog= trainData["soybean_meal"],exog = trainData.loc[:,covariatesSoybean_meal],order=(2,0,2)).fit(disp=0)
f,ax=plt.subplots(1,3,figsize=(18,8))

modelArimaCorn.plot_predict(ax=ax[0])

modelArimaSoybeans.plot_predict(ax=ax[1])

modelArimaSoybean_meal.plot_predict(ax=ax[2])

plt.show()
forecastData["corn"] = modelArimaCorn.forecast(11,exog=forecastDataThirdScheme.loc[:,covariatesCorn1])[0]

forecastData["soybeans"] = modelArimaSoybeans.forecast(11,exog=forecastDataThirdScheme.loc[:,covariatesSoybeans1])[0]

forecastData["soybean_meal"]= modelArimaSoybean_meal.forecast(11,exog=forecastData.loc[:,covariatesSoybean_meal])[0]
df = trainData.append(forecastData).loc[:,["corn","soybeans","soybean_meal"]]

Corn_tsPlot = px.line(df.reset_index(), x="year", y="corn",

              title="Total annual exports and expected from Brazil (all states and to everywhere) of corn ")

SoybeanMeal_tsPlot = px.line(df.reset_index(), x="year", y="soybean_meal",

              title="Total annual exports and expected from Brazil (all states and to everywhere) of soybean meal ")

Soybeans_tsPlot = px.line(df.reset_index(), x="year", y="soybeans",

              title="Total annual exports and expected from Brazil (all states and to everywhere) of soybeans ")



Corn_tsPlot.update_layout(shapes=[

    dict(

      type= 'line',

      yref= 'paper', y0= 0, y1= 1,

      xref= 'x', x0= 2020, x1= 2020

    )

])

SoybeanMeal_tsPlot.update_layout(shapes=[

    dict(

      type= 'line',

      yref= 'paper', y0= 0, y1= 1,

      xref= 'x', x0= 2020, x1= 2020

    )

])

Soybeans_tsPlot.update_layout(shapes=[

    dict(

      type= 'line',

      yref= 'paper', y0= 0, y1= 1,

      xref= 'x', x0= 2020, x1= 2020

    )

])

Corn_tsPlot.show()

SoybeanMeal_tsPlot.show()

Soybeans_tsPlot.show()