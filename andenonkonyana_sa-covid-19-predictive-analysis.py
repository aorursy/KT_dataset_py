#Importing required libraries for visualisation and data processing

import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from fbprophet.plot import plot_plotly, add_changepoints_to_plot

import plotly.offline as py

from datetime import date, timedelta

from statsmodels.tsa.arima_model import ARIMA

from sklearn.cluster import KMeans

from fbprophet import Prophet

import plotly.offline as py

import plotly.express as px
#Before doing analysis, I would like to quickly visualise the Symptoms of Coronavirus

#The information is taken from WHO.com



symptoms={'symptom':['Fever',

        'Dry cough',

        'Fatigue',

        'Sputum production',

        'Shortness of breath',

        'Muscle pain',

        'Sore throat',

        'Headache',

        'Chills',

        'Nausea or vomiting',

        'Nasal congestion',

        'Diarrhoea',

        'Haemoptysis',

        'Conjunctival congestion'],'percentage':[87.9,67.7,38.1,33.4,18.6,14.8,13.9,13.6,11.4,5.0,4.8,3.7,0.9,0.8]}



symptoms=pd.DataFrame(data=symptoms,index=range(14))

symptoms
#Bar Plot for visualisation



fig = px.bar(symptoms[['symptom', 'percentage']].sort_values('percentage', ascending=False), 

             y="percentage", x="symptom", color='symptom', 

             log_y=True, template='ggplot2', title='Symptoms of  Coronavirus')

fig.show()
#Pie chart showing symptoms in percentages(%)



plt.figure(figsize=(15,15))

plt.title('Symptoms of Coronavirus',fontsize=20)    

plt.pie(symptoms['percentage'],autopct='%1.1f%%')

plt.legend(symptoms['symptom'],loc='best')

plt.show()
#WordCloud visualisation



from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in symptoms.symptom)

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="white").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#Fever, Dry cough and fatigue are the most symptoms f coronavirus,

#If you develop one of these, seek medical attention or Stay AT Home!!!
data1=pd.read_csv('../input/sa-covid-dataset/SA_Data.csv')

data1.tail(10)
#Data processing



data1.describe().T
data1.shape
data1.isna().sum()
#the dataset is clean, no missing values or whatsover
#Slicing data so that I can be able to do clear visualisations



SA_Data=data1.tail(20)

SA_Data.tail()
#Map of SA that clear shows the increase of cases from the first day of announcement (March,03)

#Spread total cases over time



cmap1 = data1

cmap1  = cmap1.groupby(['Date','Lat','Long'])['Total_Cases', 'Total_Deaths', 'Total_Recoveries'].max()





cmap1 = cmap1.reset_index()

cmap1.head()

cmap1['size'] = cmap1['Total_Cases']*90000000

cmap1

fig = px.scatter_mapbox(cmap1, lat="Lat", lon="Long",

                     color="Total_Cases", size='size',hover_data=['Total_Cases','Total_Recoveries','Total_Deaths'],

                     color_continuous_scale='burgyl',

                     animation_frame="Date", 

                     title='Spread total cases over time')

fig.update(layout_coloraxis_showscale=True)

fig.update_layout(mapbox_style="carto-positron",

                  mapbox_zoom=3)

fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

fig.show()
#Visualisation of confimerd cases,Recoveries and Death

#Total confirmed cases vs Recoveries vs Death Cases



plt.figure(figsize=(23,10))

plt.bar(SA_Data.Date, SA_Data.Total_Cases,label="Confirm")

plt.xlabel('Date')

plt.ylabel("Count")

plt.legend(frameon=True, fontsize=12)

plt.title("Confirmation",fontsize=50)

plt.show()



plt.figure(figsize=(23,10))

plt.bar(SA_Data.Date, SA_Data.Total_Recoveries,label="Recovery")

plt.xlabel('Date')

plt.ylabel("Count")

plt.legend(frameon=True, fontsize=12)

plt.title("Recoveries",fontsize=50)

plt.show()



plt.figure(figsize=(23,10))

plt.bar(SA_Data.Date, SA_Data.Total_Deaths,label="Death")

plt.xlabel('Date')

plt.ylabel("Count")

plt.legend(frameon=True, fontsize=12)

plt.title("Death Cases",fontsize=50)

plt.show()
#From the graphs above, we can see that the number of all three cases are rapidly increasing,

#Even though,there is consistency on the recovery cases some days
#Subplots showing three cases



plt.figure(figsize=(23,10))

plt.bar(SA_Data.Date, SA_Data.Total_Cases,label="Confirmed Cases")

plt.bar(SA_Data.Date, SA_Data.Total_Recoveries,label="Total_Recoveries")

plt.bar(SA_Data.Date, SA_Data.Total_Deaths,label="Total_Deaths")

plt.xlabel('Date')

plt.ylabel("Count")

plt.legend(frameon=True, fontsize=12)

plt.title("Confirmation vs Recoveries vs Death",fontsize=50)

plt.show()



f, ax = plt.subplots(figsize=(23,10))

ax=sns.scatterplot(x="Date", y="Total_Cases", data=SA_Data,

             color="black",label = "Confirm")

ax=sns.scatterplot(x="Date", y="Total_Recoveries", data=SA_Data,

             color="red",label = "Recovery")

ax=sns.scatterplot(x="Date", y="Total_Deaths", data=SA_Data,

             color="blue",label = "Total_Deaths")

plt.plot(SA_Data.Date,SA_Data.Total_Cases,zorder=1,color="black")

plt.plot(SA_Data.Date,SA_Data.Total_Recoveries,zorder=1,color="red")

plt.plot(SA_Data.Date,SA_Data.Total_Deaths,zorder=1,color="blue")
#Total Cases today



daily = data1.sort_values(['Date'])

latest = data1[data1.Date == daily.Date.max()]

latest.head()
#Renaming columns for when I do predictive analysis



data=latest.rename(columns={ "Total_Cases":"Confirmed","Total_Deaths": "Death","Total_Recoveries":"Recovered"})

data.head()
#Grouping by date



dgd=data.groupby("Date")[['Confirmed', 'Death', 'Recovered']].sum().reset_index()



dgd.head()
#Ratio and Percentage of Recovery and Death after Confirmation

#r_cm= ratio of comfirmed cases

#d_cm= ratio of death cases



r_cm = float(dgd.Recovered/dgd.Confirmed)

d_cm = float(dgd.Death/dgd.Confirmed)

print("The percentage of recovery after confirmation is "+ str(r_cm*100) )

print("The percentage of death after confirmation is "+ str(d_cm*100) )
#Confirmed cases prediction using prophet



prophet=data1.iloc[: , [0,1 ]].copy() 

prophet.head()

prophet.columns = ['ds','y']

prophet.tail()
m=Prophet()

m.fit(prophet)

future=m.make_future_dataframe(periods=15)

forecast=m.predict(future)

forecast
#Predicted values/Cases



cnfrm = forecast.loc[:,['ds','trend']]

cnfrm = cnfrm[cnfrm['trend']>0]

cnfrm=cnfrm.tail(15)

cnfrm.columns = ['Date','Confirmed']

cnfrm.head()
#Graphical Representation of Predicted Confirmed cases



figure = plot_plotly(m, forecast)

py.iplot(figure) 



figure = m.plot(forecast,xlabel='Date',ylabel='Confirmed Count')
#Prediction of Recoveries



prophet_rec=data1.iloc[: , [0,3 ]].copy() 

prophet_rec.head()

prophet_rec.columns = ['ds','y']

prophet_rec.tail()
m1=Prophet()

m1.fit(prophet_rec)

future_rec=m1.make_future_dataframe(periods=15)

forecast_rec=m1.predict(future_rec)

forecast_rec
rec = forecast_rec.loc[:,['ds','trend']]

rec = rec[rec['trend']>0]

rec=rec.tail(15)

rec.columns = ['Date','Recovery']

rec.head()
#Graphical Representation of Predicted Recoveries



figure_rec = plot_plotly(m1, forecast_rec)

py.iplot(figure_rec) 



figure_rec = m1.plot(forecast_rec,xlabel='Date',ylabel='Recovery Count')
prophet_dth=data1.iloc[: , [0,2 ]].copy() 

prophet_dth.head()

prophet_dth.columns = ['ds','y']

prophet_dth.tail()
m2=Prophet()

m2.fit(prophet_dth)

future_dth=m2.make_future_dataframe(periods=15)

forecast_dth=m2.predict(future_dth)

forecast_dth
dth = forecast_dth.loc[:,['ds','trend']]

dth = dth[dth['trend']>0]

dth=dth.tail(15)

dth.columns = ['Date','Death']

dth.head()
##Graphical Representation of Predicted Death Cases



figure_dth = plot_plotly(m2, forecast_dth)

py.iplot(figure_dth) 



figure_dth = m2.plot(forecast_dth,xlabel='Date',ylabel='Death Count')
#Combined Predictions



prediction = cnfrm

prediction['Recover'] = rec.Recovery

prediction['Death'] = dth.Death

prediction.head()
#From above, it can be seen that the number of confirmed and death cases will continue to rise

#Recoveries will at times lag as the number of infected people will also continue to recover

#This prediction is truly based on the dataset depend on the current situation. 

#If we able to get vaccine there will be gradual changes in recovery or even confirmed and death cases.

#The solution for now is to stay at home, wear masks, sanitize.