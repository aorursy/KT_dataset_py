import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

!pip install plotly

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import numpy as np

import datetime as dt

from datetime import timedelta

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score,silhouette_samples

from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error,r2_score

import statsmodels.api as sm

from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing

from fbprophet import Prophet

from sklearn.preprocessing import PolynomialFeatures

from statsmodels.tsa.stattools import adfuller

!pip install pyramid-arima

from pyramid.arima import auto_arima

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

std=StandardScaler()

#pd.set_option('display.float_format', lambda x: '%.6f' % x)
covid=pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

covid.head()
print("Size/Shape of the dataset: ",covid.shape)

print("Checking for null values:\n",covid.isnull().sum())

print("Checking Data-type of each column:\n",covid.dtypes)
#Dropping column as SNo is of no use, and "Province/State" contains too many missing values

covid.drop(["SNo"],1,inplace=True)
#Converting "Observation Date" into Datetime format

covid["ObservationDate"]=pd.to_datetime(covid["ObservationDate"])
grouped_country=covid.groupby(["Country/Region","ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
grouped_country["Active Cases"]=grouped_country["Confirmed"]-grouped_country["Recovered"]-grouped_country["Deaths"]

grouped_country["log_confirmed"]=np.log(grouped_country["Confirmed"])

grouped_country["log_active"]=np.log(grouped_country["Active Cases"])
#Grouping different types of cases as per the date

datewise=covid.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise["Days Since"]=datewise.index-datewise.index.min()
print("Basic Information")

print("Totol number of countries with Disease Spread: ",len(covid["Country/Region"].unique()))

print("Total number of Confirmed Cases around the World: ",datewise["Confirmed"].iloc[-1])

print("Total number of Recovered Cases around the World: ",datewise["Recovered"].iloc[-1])

print("Total number of Deaths Cases around the World: ",datewise["Deaths"].iloc[-1])

print("Total number of Active Cases around the World: ",(datewise["Confirmed"].iloc[-1]-datewise["Recovered"].iloc[-1]-datewise["Deaths"].iloc[-1]))

print("Total number of Closed Cases around the World: ",datewise["Recovered"].iloc[-1]+datewise["Deaths"].iloc[-1])

print("Approximate number of Confirmed Cases per Day around the World: ",np.round(datewise["Confirmed"].iloc[-1]/datewise.shape[0]))

print("Approximate number of Recovered Cases per Day around the World: ",np.round(datewise["Recovered"].iloc[-1]/datewise.shape[0]))

print("Approximate number of Death Cases per Day around the World: ",np.round(datewise["Deaths"].iloc[-1]/datewise.shape[0]))

print("Approximate number of Confirmed Cases per hour around the World: ",np.round(datewise["Confirmed"].iloc[-1]/((datewise.shape[0])*24)))

print("Approximate number of Recovered Cases per hour around the World: ",np.round(datewise["Recovered"].iloc[-1]/((datewise.shape[0])*24)))

print("Approximate number of Death Cases per hour around the World: ",np.round(datewise["Deaths"].iloc[-1]/((datewise.shape[0])*24)))

print("Number of Confirmed Cases in last 24 hours: ",datewise["Confirmed"].iloc[-1]-datewise["Confirmed"].iloc[-2])

print("Number of Recovered Cases in last 24 hours: ",datewise["Recovered"].iloc[-1]-datewise["Recovered"].iloc[-2])

print("Number of Death Cases in last 24 hours: ",datewise["Deaths"].iloc[-1]-datewise["Deaths"].iloc[-2])
fig=px.bar(x=datewise.index,y=datewise["Confirmed"]-datewise["Recovered"]-datewise["Deaths"])

fig.update_layout(title="Distribution of Number of Active Cases",

                  xaxis_title="Date",yaxis_title="Number of Cases",)

fig.show()
fig=px.bar(x=datewise.index,y=datewise["Recovered"]+datewise["Deaths"])

fig.update_layout(title="Distribution of Number of Closed Cases",

                  xaxis_title="Date",yaxis_title="Number of Cases")

fig.show()
datewise["WeekOfYear"]=datewise.index.weekofyear



week_num=[]

weekwise_confirmed=[]

weekwise_recovered=[]

weekwise_deaths=[]

w=1

for i in list(datewise["WeekOfYear"].unique()):

    weekwise_confirmed.append(datewise[datewise["WeekOfYear"]==i]["Confirmed"].iloc[-1])

    weekwise_recovered.append(datewise[datewise["WeekOfYear"]==i]["Recovered"].iloc[-1])

    weekwise_deaths.append(datewise[datewise["WeekOfYear"]==i]["Deaths"].iloc[-1])

    week_num.append(w)

    w=w+1



fig=go.Figure()

fig.add_trace(go.Scatter(x=week_num, y=weekwise_confirmed,

                    mode='lines+markers',

                    name='Weekly Growth of Confirmed Cases'))

fig.add_trace(go.Scatter(x=week_num, y=weekwise_recovered,

                    mode='lines+markers',

                    name='Weekly Growth of Recovered Cases'))

fig.add_trace(go.Scatter(x=week_num, y=weekwise_deaths,

                    mode='lines+markers',

                    name='Weekly Growth of Death Cases'))

fig.update_layout(title="Weekly Growth of different types of Cases in India",

                 xaxis_title="Week Number",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(15,5))

sns.barplot(x=week_num,y=pd.Series(weekwise_confirmed).diff().fillna(0),ax=ax1)

sns.barplot(x=week_num,y=pd.Series(weekwise_deaths).diff().fillna(0),ax=ax2)

ax1.set_xlabel("Week Number")

ax2.set_xlabel("Week Number")

ax1.set_ylabel("Number of Confirmed Cases")

ax2.set_ylabel("Number of Death Cases")

ax1.set_title("Weekly increase in Number of Confirmed Cases")

ax2.set_title("Weekly increase in Number of Death Cases")
fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Confirmed"],

                    mode='lines+markers',

                    name='Confirmed Cases'))

fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Recovered"],

                    mode='lines+markers',

                    name='Recovered Cases'))

fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Deaths"],

                    mode='lines+markers',

                    name='Death Cases'))

fig.update_layout(title="Growth of different types of cases",

                 xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
#Calculating the Mortality Rate and Recovery Rate

datewise["Mortality Rate"]=(datewise["Deaths"]/datewise["Confirmed"])*100

datewise["Recovery Rate"]=(datewise["Recovered"]/datewise["Confirmed"])*100

datewise["Active Cases"]=datewise["Confirmed"]-datewise["Recovered"]-datewise["Deaths"]

datewise["Closed Cases"]=datewise["Recovered"]+datewise["Deaths"]



print("Average Mortality Rate",datewise["Mortality Rate"].mean())

print("Median Mortality Rate",datewise["Mortality Rate"].median())

print("Average Recovery Rate",datewise["Recovery Rate"].mean())

print("Median Recovery Rate",datewise["Recovery Rate"].median())



#Plotting Mortality and Recovery Rate 

fig = make_subplots(rows=2, cols=1,

                   subplot_titles=("Recovery Rate", "Mortatlity Rate"))

fig.add_trace(

    go.Scatter(x=datewise.index, y=(datewise["Recovered"]/datewise["Confirmed"])*100,name="Recovery Rate"),

    row=1, col=1

)

fig.add_trace(

    go.Scatter(x=datewise.index, y=(datewise["Deaths"]/datewise["Confirmed"])*100,name="Mortality Rate"),

    row=2, col=1

)

fig.update_layout(height=1000,legend=dict(x=-0.1,y=1.2,traceorder="normal"))

fig.update_xaxes(title_text="Date", row=1, col=1)

fig.update_yaxes(title_text="Recovery Rate", row=1, col=1)

fig.update_xaxes(title_text="Date", row=1, col=2)

fig.update_yaxes(title_text="Mortality Rate", row=1, col=2)

fig.show()
print("Average increase in number of Confirmed Cases every day: ",np.round(datewise["Confirmed"].diff().fillna(0).mean()))

print("Average increase in number of Recovered Cases every day: ",np.round(datewise["Recovered"].diff().fillna(0).mean()))

print("Average increase in number of Deaths Cases every day: ",np.round(datewise["Deaths"].diff().fillna(0).mean()))



fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Confirmed"].diff().fillna(0),mode='lines+markers',

                    name='Confirmed Cases'))

fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Recovered"].diff().fillna(0),mode='lines+markers',

                    name='Recovered Cases'))

fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Deaths"].diff().fillna(0),mode='lines+markers',

                    name='Death Cases'))

fig.update_layout(title="Daily increase in different types of Cases",

                 xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Confirmed"].diff().rolling(window=7).mean(),mode='lines+markers',

                    name='Confirmed Cases'))

fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Recovered"].diff().rolling(window=7).mean(),mode='lines+markers',

                    name='Recovered Cases'))

fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Deaths"].diff().rolling(window=7).mean(),mode='lines+markers',

                    name='Death Cases'))

fig.update_layout(title="7 Days Rolling Mean of Daily Increase of Confirmed, Recovered and Death Cases",

                 xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
print("Average growth factor of number of Confirmed Cases: ",(datewise["Confirmed"]/datewise["Confirmed"].shift()).mean())

print("Median growth factor of number of Confirmed Cases: ",(datewise["Confirmed"]/datewise["Confirmed"].shift()).median())

print("Average growth factor of number of Recovered Cases: ",(datewise["Recovered"]/datewise["Recovered"].shift()).mean())

print("Median growth factor of number of Recovered Cases: ",(datewise["Recovered"]/datewise["Recovered"].shift()).median())

print("Average growth factor of number of Death Cases: ",(datewise["Deaths"]/datewise["Deaths"].shift()).mean())

print("Median growth factor of number of Death Cases: ",(datewise["Deaths"]/datewise["Deaths"].shift()).median())



fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Confirmed"]/datewise["Confirmed"].shift(),

                    mode='lines',

                    name='Growth Factor of Confirmed Cases'))

fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Recovered"]/datewise["Recovered"].shift(),

                    mode='lines',

                    name='Growth Factor of Recovered Cases'))

fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Deaths"]/datewise["Deaths"].shift(),

                    mode='lines',

                    name='Growth Factor of Death Cases'))

fig.update_layout(title="Datewise Growth Factor of different types of cases",

                 xaxis_title="Date",yaxis_title="Growth Factor",

                 legend=dict(x=0,y=-0.4,traceorder="normal"))

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise.index, 

                         y=(datewise["Confirmed"]-datewise["Recovered"]-datewise["Deaths"])/(datewise["Confirmed"]-datewise["Recovered"]-datewise["Deaths"]).shift(),

                    mode='lines',

                    name='Growth Factor of Active Cases'))

fig.add_trace(go.Scatter(x=datewise.index, y=(datewise["Recovered"]+datewise["Deaths"])/(datewise["Recovered"]+datewise["Deaths"]).shift(),

                    mode='lines',

                    name='Growth Factor of Closed Cases'))

fig.update_layout(title="Datewise Growth Factor of Active and Closed Cases",

                 xaxis_title="Date",yaxis_title="Growth Factor",

                 legend=dict(x=0,y=-0.4,traceorder="normal"))

fig.show()
c=570

double_days=[]

C=[]

while(1):

    double_days.append(datewise[datewise["Confirmed"]<=c].iloc[[-1]]["Days Since"][0])

    C.append(c)

    c=c*2

    if(c<datewise["Confirmed"].max()):

        continue

    else:

        break
doubling_rate=pd.DataFrame(list(zip(C,double_days)),columns=["No. of cases","Days since first Case"])

doubling_rate["Number of days for doubling"]=doubling_rate["Days since first Case"].diff().fillna(doubling_rate["Days since first Case"])

doubling_rate
c1=100000

days_200k=[]

C1=[]

while(1):

    days_200k.append(datewise[datewise["Confirmed"]<=c1].iloc[[-1]]["Days Since"][0])

    C1.append(c1)

    c1=c1+200000

    if(c1<datewise["Confirmed"].max()):

        continue

    else:

        break
rate_200k=pd.DataFrame(list(zip(C1,days_200k)),columns=["No. of Cases","Days Since first Case"])

rate_200k["Days requried for rise of 200K"]=rate_200k["Days Since first Case"].diff().fillna(rate_200k["Days Since first Case"].iloc[[0]][0])



fig=go.Figure()

fig.add_trace(go.Scatter(x=rate_200k["No. of Cases"], y=rate_200k["Days requried for rise of 200K"].dt.days,

                    mode='lines+markers',

                    name='Weekly Growth of Confirmed Cases'))

fig.update_layout(title="Number of Days required for increase in number of cases by 200K",

                 xaxis_title="Number of Cases",yaxis_title="Number of Days")

fig.show()
#Calculating countrywise Moratality and Recovery Rate

countrywise=covid[covid["ObservationDate"]==covid["ObservationDate"].max()].groupby(["Country/Region"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'}).sort_values(["Confirmed"],ascending=False)

countrywise["Mortality"]=(countrywise["Deaths"]/countrywise["Confirmed"])*100

countrywise["Recovery"]=(countrywise["Recovered"]/countrywise["Confirmed"])*100
country_last_24_confirmed=[]

country_last_24_recovered=[]

country_last_24_deaths=[]

for country in countrywise.index:

    country_last_24_confirmed.append((grouped_country.loc[country].iloc[-1]-grouped_country.loc[country].iloc[-2])["Confirmed"])

    country_last_24_recovered.append((grouped_country.loc[country].iloc[-1]-grouped_country.loc[country].iloc[-2])["Recovered"])

    country_last_24_deaths.append((grouped_country.loc[country].iloc[-1]-grouped_country.loc[country].iloc[-2])["Deaths"])
Last_24_Hours_country=pd.DataFrame(list(zip(countrywise.index,country_last_24_confirmed,country_last_24_recovered,country_last_24_deaths)),

                                   columns=["Country Name","Last 24 Hours Confirmed","Last 24 Hours Recovered","Last 24 Hours Deaths"])
Top_15_Confirmed_24hr=Last_24_Hours_country.sort_values(["Last 24 Hours Confirmed"],ascending=False).head(15)

Top_15_Recoverd_24hr=Last_24_Hours_country.sort_values(["Last 24 Hours Recovered"],ascending=False).head(15)

Top_15_Deaths_24hr=Last_24_Hours_country.sort_values(["Last 24 Hours Deaths"],ascending=False).head(15)





fig, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize=(10,20))

sns.barplot(x=Top_15_Confirmed_24hr["Last 24 Hours Confirmed"],y=Top_15_Confirmed_24hr["Country Name"],ax=ax1)

ax1.set_title("Top 15 Countries with Highest Number of Confirmed Cases in Last 24 Hours")

sns.barplot(x=Top_15_Recoverd_24hr["Last 24 Hours Recovered"],y=Top_15_Recoverd_24hr["Country Name"],ax=ax2)

ax2.set_title("Top 15 Countries with Highest Number of Recovered Cases in Last 24 Hours")

sns.barplot(x=Top_15_Deaths_24hr["Last 24 Hours Deaths"],y=Top_15_Deaths_24hr["Country Name"],ax=ax3)

ax3.set_title("Top 15 Countries with Highest Number of Death Cases in Last 24 Hours")
Last_24_Hours_country["Proportion of Confirmed"]=(Last_24_Hours_country["Last 24 Hours Confirmed"]/(datewise["Confirmed"].iloc[-1]-datewise["Confirmed"].iloc[-2]))*100

Last_24_Hours_country["Proportion of Recovered"]=(Last_24_Hours_country["Last 24 Hours Recovered"]/(datewise["Recovered"].iloc[-1]-datewise["Recovered"].iloc[-2]))*100

Last_24_Hours_country["Proportion of Deaths"]=(Last_24_Hours_country["Last 24 Hours Deaths"]/(datewise["Deaths"].iloc[-1]-datewise["Deaths"].iloc[-2]))*100
Last_24_Hours_country[["Country Name","Proportion of Confirmed","Proportion of Recovered","Proportion of Deaths"]].sort_values(["Proportion of Confirmed"],ascending=False).style.background_gradient(cmap="Reds")
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10,12))

top_15_confirmed=countrywise.sort_values(["Confirmed"],ascending=False).head(15)

top_15_deaths=countrywise.sort_values(["Deaths"],ascending=False).head(15)

sns.barplot(x=top_15_confirmed["Confirmed"],y=top_15_confirmed.index,ax=ax1)

ax1.set_title("Top 15 countries as per Number of Confirmed Cases")

sns.barplot(x=top_15_deaths["Deaths"],y=top_15_deaths.index,ax=ax2)

ax2.set_title("Top 15 countries as per Number of Death Cases")
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10,15))

countrywise_plot_mortal=countrywise[countrywise["Confirmed"]>500].sort_values(["Mortality"],ascending=False).head(15)

sns.barplot(x=countrywise_plot_mortal["Mortality"],y=countrywise_plot_mortal.index,ax=ax1)

ax1.set_title("Top 15 Countries according High Mortatlity Rate")

ax1.set_xlabel("Mortality (in Percentage)")

countrywise_plot_recover=countrywise[countrywise["Confirmed"]>500].sort_values(["Recovery"],ascending=False).head(15)

sns.barplot(x=countrywise_plot_recover["Recovery"],y=countrywise_plot_recover.index, ax=ax2)

ax2.set_title("Top 15 Countries according High Recovery Rate")

ax2.set_xlabel("Recovery (in Percentage)")
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10,15))

countrywise_plot_mortal=countrywise[countrywise["Confirmed"]>500].sort_values(["Mortality"],ascending=False).tail(15)

sns.barplot(x=countrywise_plot_mortal["Mortality"],y=countrywise_plot_mortal.index,ax=ax1)

ax1.set_title("Top 15 Countries according Low Mortatlity Rate")

ax1.set_xlabel("Mortality (in Percentage)")

countrywise_plot_recover=countrywise[countrywise["Confirmed"]>500].sort_values(["Recovery"],ascending=False).tail(15)

sns.barplot(x=countrywise_plot_recover["Recovery"],y=countrywise_plot_recover.index, ax=ax2)

ax2.set_title("Top 15 Countries according Low Recovery Rate")

ax2.set_xlabel("Recovery (in Percentage)")
no_recovered_countries=countrywise[(countrywise["Recovered"]==0)][["Confirmed","Deaths"]]

no_recovered_countries["Mortality Rate"]=(no_recovered_countries["Deaths"]/no_recovered_countries["Confirmed"])*100

no_recovered_countries=no_recovered_countries[no_recovered_countries["Mortality Rate"]>0].sort_values(["Mortality Rate"],ascending=False)

no_recovered_countries.style.background_gradient('Reds')
no_deaths=countrywise[(countrywise["Confirmed"]>100)&(countrywise["Deaths"]==0)]

no_deaths=no_deaths[no_deaths["Recovery"]>0].sort_values(["Recovery"],ascending=False).drop(["Mortality"],1)

no_deaths.style.background_gradient(cmap="Reds")
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10,15))

countrywise["Active Cases"]=(countrywise["Confirmed"]-countrywise["Recovered"]-countrywise["Deaths"])

countrywise["Outcome Cases"]=(countrywise["Recovered"]+countrywise["Deaths"])

top_15_active=countrywise.sort_values(["Active Cases"],ascending=False).head(15)

top_15_outcome=countrywise.sort_values(["Outcome Cases"],ascending=False).head(15)

sns.barplot(x=top_15_active["Active Cases"],y=top_15_active.index,ax=ax1)

sns.barplot(x=top_15_outcome["Outcome Cases"],y=top_15_outcome.index,ax=ax2)

ax1.set_title("Top 15 Countries with Most Number of Active Cases")

ax2.set_title("Top 15 Countries with Most Number of Closed Cases")
country_date=covid.groupby(["Country/Region","ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

confirm_rate=[]

for country in countrywise.index:

    days=country_date.ix[country].shape[0]

    confirm_rate.append((countrywise.ix[country]["Confirmed"])/days)

countrywise["Confirm Cases/Day"]=confirm_rate
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10,15))

top_15_ccpd=countrywise.sort_values(["Confirm Cases/Day"],ascending=False).head(15)

sns.barplot(y=top_15_ccpd.index,x=top_15_ccpd["Confirm Cases/Day"],ax=ax1)

ax1.set_title("Top 15 countries as per high number Confirmed Cases per Day")

bottom_15_ccpd=countrywise[countrywise["Confirmed"]>1000].sort_values(["Confirm Cases/Day"],ascending=False).tail(15)

sns.barplot(y=bottom_15_ccpd.index,x=bottom_15_ccpd["Confirm Cases/Day"],ax=ax2)

ax2.set_title("Top 15 countries as per Lowest Confirmed Cases per Day having more than 1000 Confirmed Cases")
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10,15))

countrywise["Survival Probability"]=(1-(countrywise["Deaths"]/countrywise["Confirmed"]))*100

top_25_survival=countrywise[countrywise["Confirmed"]>1000].sort_values(["Survival Probability"],ascending=False).head(15)

sns.barplot(x=top_25_survival["Survival Probability"],y=top_25_survival.index,ax=ax1)

ax1.set_title("Top 25 Countries with Maximum Survival Probability having more than 1000 Confiremed Cases")

print('Mean Survival Probability across all countries',countrywise["Survival Probability"].mean())

print('Median Survival Probability across all countries',countrywise["Survival Probability"].median())

print('Mean Death Probability across all countries',100-countrywise["Survival Probability"].mean())

print('Median Death Probability across all countries',100-countrywise["Survival Probability"].median())



Bottom_5_countries=countrywise[countrywise["Confirmed"]>100].sort_values(["Survival Probability"],ascending=True).head(15)

sns.barplot(x=Bottom_5_countries["Survival Probability"],y=Bottom_5_countries.index,ax=ax2)

plt.title("Bottom 15 Countries as per Survival Probability")
fig=go.Figure()

for country in countrywise.head(10).index:

    fig.add_trace(go.Scatter(x=grouped_country.ix[country]["log_confirmed"], y=grouped_country.ix[country]["log_active"],

                    mode='lines',name=country))

fig.update_layout(height=600,title="COVID-19 Journey of some worst affected countries and India",

                 xaxis_title="Confirmed Cases (Logrithmic Scale)",yaxis_title="Active Cases (Logarithmic Scale)",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

for country in countrywise.head(10).index:

    fig.add_trace(go.Scatter(x=grouped_country.ix[country].index, y=grouped_country.ix[country]["Confirmed"].rolling(window=7).mean().diff(),

                    mode='lines',name=country))

fig.update_layout(height=600,title="7 Days Rolling Average of Daily increase of Confirmed Cases",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

for country in countrywise.head(10).index:

    fig.add_trace(go.Scatter(x=grouped_country.ix[country].index, 

                             y=grouped_country.ix[country]["Deaths"].rolling(window=7).mean().diff(),

                    mode='lines',name=country))

fig.update_layout(height=600,title="7 Days Rolling Average of Daily increase of Death Cases",

                 xaxis_title="Date",yaxis_title="Death Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

for country in countrywise.head(10).index:

    fig.add_trace(go.Scatter(x=grouped_country.ix[country].index, 

                             y=grouped_country.ix[country]["Recovered"].rolling(window=7).mean().diff(),

                    mode='lines',name=country))

fig.update_layout(height=600,title="7 Days Rolling Average of Daily increase of Recovered Cases",

                 xaxis_title="Date",yaxis_title="Recovered Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
X=countrywise[["Mortality","Recovery"]]

#Standard Scaling since K-Means Clustering is a distance based alogrithm

X=std.fit_transform(X) 
wcss=[]

sil=[]

for i in range(2,11):

    clf=KMeans(n_clusters=i,init='k-means++',random_state=42)

    clf.fit(X)

    labels=clf.labels_

    centroids=clf.cluster_centers_

    sil.append(silhouette_score(X, labels, metric='euclidean'))

    wcss.append(clf.inertia_)
x=np.arange(2,11)

plt.figure(figsize=(10,5))

plt.plot(x,wcss,marker='o')

plt.xlabel("Number of Clusters")

plt.ylabel("Within Cluster Sum of Squares (WCSS)")

plt.title("Elbow Method")
import scipy.cluster.hierarchy as sch

plt.figure(figsize=(20,15))

dendogram=sch.dendrogram(sch.linkage(X, method  = "ward"))
clf_final=KMeans(n_clusters=3,init='k-means++',random_state=6)

clf_final.fit(X)
countrywise["Clusters"]=clf_final.predict(X)
cluster_summary=pd.concat([countrywise[countrywise["Clusters"]==1].head(15),countrywise[countrywise["Clusters"]==2].head(15),countrywise[countrywise["Clusters"]==0].head(15)])

cluster_summary.style.background_gradient(cmap='Reds').format("{:.2f}")
print("Avergae Mortality Rate of Cluster 0: ",countrywise[countrywise["Clusters"]==0]["Mortality"].mean())

print("Avergae Recovery Rate of Cluster 0: ",countrywise[countrywise["Clusters"]==0]["Recovery"].mean())

print("Avergae Mortality Rate of Cluster 1: ",countrywise[countrywise["Clusters"]==1]["Mortality"].mean())

print("Avergae Recovery Rate of Cluster 1: ",countrywise[countrywise["Clusters"]==1]["Recovery"].mean())

print("Avergae Mortality Rate of Cluster 2: ",countrywise[countrywise["Clusters"]==2]["Mortality"].mean())

print("Avergae Recovery Rate of Cluster 2: ",countrywise[countrywise["Clusters"]==2]["Recovery"].mean())
plt.figure(figsize=(10,5))

sns.scatterplot(x=countrywise["Recovery"],y=countrywise["Mortality"],hue=countrywise["Clusters"],s=100)

plt.axvline(((datewise["Recovered"]/datewise["Confirmed"])*100).mean(),

            color='red',linestyle="--",label="Mean Recovery Rate around the World")

plt.axhline(((datewise["Deaths"]/datewise["Confirmed"])*100).mean(),

            color='black',linestyle="--",label="Mean Mortality Rate around the World")

plt.legend()
print("Few Countries belonging to Cluster 0: ",list(countrywise[countrywise["Clusters"]==0].head(10).index))

print("Few Countries belonging to Cluster 1: ",list(countrywise[countrywise["Clusters"]==1].head(10).index))

print("Few Countries belonging to Cluster 2: ",list(countrywise[countrywise["Clusters"]==2].head(10).index))
china_data=covid[covid["Country/Region"]=="Mainland China"]

Italy_data=covid[covid["Country/Region"]=="Italy"]

US_data=covid[covid["Country/Region"]=="US"]

spain_data=covid[covid["Country/Region"]=="Spain"]

brazil_data=covid[covid["Country/Region"]=="Brazil"]

rest_of_world=covid[(covid["Country/Region"]!="Mainland China")&(covid["Country/Region"]!="Italy")&(covid["Country/Region"]!="US")&(covid["Country/Region"]!="Spain")&(covid["Country/Region"]!="Brazil")]



datewise_china=china_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise_Italy=Italy_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise_US=US_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise_Spain=spain_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise_Brazil=brazil_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise_restofworld=rest_of_world.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise_china.index, y=(datewise_china["Confirmed"]),

                    mode='lines',name="China"))

fig.add_trace(go.Scatter(x=datewise_Italy.index, y=(datewise_Italy["Confirmed"]),

                    mode='lines',name="Italy"))

fig.add_trace(go.Scatter(x=datewise_US.index, y=(datewise_US["Confirmed"]),

                    mode='lines',name="United States"))

fig.add_trace(go.Scatter(x=datewise_Spain.index, y=(datewise_Spain["Confirmed"]),

                    mode='lines',name="Spain"))

fig.add_trace(go.Scatter(x=datewise_Brazil.index, y=(datewise_Brazil["Confirmed"]),

                    mode='lines',name="Brazil"))

fig.add_trace(go.Scatter(x=datewise_restofworld.index, y=(datewise_restofworld["Confirmed"]),

                    mode='lines',name="Rest of the World"))

fig.update_layout(title="Confirmed Cases plot",

                  xaxis_title="Date",yaxis_title="Number of Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise_china.index, y=(datewise_china["Recovered"]),

                    mode='lines',name="China"))

fig.add_trace(go.Scatter(x=datewise_Italy.index, y=(datewise_Italy["Recovered"]),

                    mode='lines',name="Italy"))

fig.add_trace(go.Scatter(x=datewise_US.index, y=(datewise_US["Recovered"]),

                    mode='lines',name="United States"))

fig.add_trace(go.Scatter(x=datewise_Spain.index, y=(datewise_Spain["Recovered"]),

                    mode='lines',name="Spain"))

fig.add_trace(go.Scatter(x=datewise_Brazil.index, y=(datewise_Brazil["Recovered"]),

                    mode='lines',name="Brazil"))

fig.add_trace(go.Scatter(x=datewise_restofworld.index, y=(datewise_restofworld["Recovered"]),

                    mode='lines',name="Rest of the World"))

fig.update_layout(title="Recovered Cases plot",

                  xaxis_title="Date",yaxis_title="Number of Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise_china.index, y=(datewise_china["Deaths"]),

                    mode='lines',name="China"))

fig.add_trace(go.Scatter(x=datewise_Italy.index, y=(datewise_Italy["Deaths"]),

                    mode='lines',name="Italy"))

fig.add_trace(go.Scatter(x=datewise_US.index, y=(datewise_US["Deaths"]),

                    mode='lines',name="United States"))

fig.add_trace(go.Scatter(x=datewise_Spain.index, y=(datewise_Spain["Deaths"]),

                    mode='lines',name="Spain"))

fig.add_trace(go.Scatter(x=datewise_Brazil.index, y=(datewise_Brazil["Deaths"]),

                    mode='lines',name="Brazil"))

fig.add_trace(go.Scatter(x=datewise_restofworld.index, y=(datewise_restofworld["Deaths"]),

                    mode='lines',name="Rest of the World"))

fig.update_layout(title="Death Cases plot",

                  xaxis_title="Date",yaxis_title="Number of Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
datewise_china["Mortality"]=(datewise_china["Deaths"]/datewise_china["Confirmed"])*100

datewise_Italy["Mortality"]=(datewise_Italy["Deaths"]/datewise_Italy["Confirmed"])*100

datewise_US["Mortality"]=(datewise_US["Deaths"]/datewise_US["Confirmed"])*100

datewise_Spain["Mortality"]=(datewise_Spain["Deaths"]/datewise_Spain["Confirmed"])*100

datewise_Brazil["Mortality"]=(datewise_Brazil["Deaths"]/datewise_Brazil["Confirmed"])*100

datewise_restofworld["Mortality"]=(datewise_restofworld["Deaths"]/datewise_restofworld["Confirmed"])*100



datewise_china["Recovery"]=(datewise_china["Recovered"]/datewise_china["Confirmed"])*100

datewise_Italy["Recovery"]=(datewise_Italy["Recovered"]/datewise_Italy["Confirmed"])*100

datewise_US["Recovery"]=(datewise_US["Recovered"]/datewise_US["Confirmed"])*100

datewise_Spain["Recovery"]=(datewise_Spain["Recovered"]/datewise_Spain["Confirmed"])*100

datewise_Brazil["Recovery"]=(datewise_Brazil["Recovered"]/datewise_Brazil["Confirmed"])*100

datewise_restofworld["Recovery"]=(datewise_restofworld["Recovered"]/datewise_restofworld["Confirmed"])*100
fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise_china.index, y=(datewise_china["Mortality"]),

                    mode='lines',name="China"))

fig.add_trace(go.Scatter(x=datewise_Italy.index, y=(datewise_Italy["Mortality"]),

                    mode='lines',name="Italy"))

fig.add_trace(go.Scatter(x=datewise_US.index, y=(datewise_US["Mortality"]),

                    mode='lines',name="United States"))

fig.add_trace(go.Scatter(x=datewise_Spain.index, y=(datewise_Spain["Mortality"]),

                    mode='lines',name="Spain"))

fig.add_trace(go.Scatter(x=datewise_Brazil.index, y=(datewise_Brazil["Mortality"]),

                    mode='lines',name="Brazil"))

fig.add_trace(go.Scatter(x=datewise_restofworld.index, y=(datewise_restofworld["Mortality"]),

                    mode='lines',name="Rest of the World"))

fig.update_layout(title="Mortality Rate comparison plot",

                  xaxis_title="Date",yaxis_title="Mortality Rate",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise_china.index, y=(datewise_china["Recovery"]),

                    mode='lines',name="China"))

fig.add_trace(go.Scatter(x=datewise_Italy.index, y=(datewise_Italy["Recovery"]),

                    mode='lines',name="Italy"))

fig.add_trace(go.Scatter(x=datewise_US.index, y=(datewise_US["Recovery"]),

                    mode='lines',name="United States"))

fig.add_trace(go.Scatter(x=datewise_Spain.index, y=(datewise_Spain["Recovery"]),

                    mode='lines',name="Spain"))

fig.add_trace(go.Scatter(x=datewise_Brazil.index, y=(datewise_Brazil["Recovery"]),

                    mode='lines',name="Brazil"))

fig.add_trace(go.Scatter(x=datewise_restofworld.index, y=(datewise_restofworld["Recovery"]),

                    mode='lines',name="Rest of the World"))

fig.update_layout(title="Recovery Rate comparison plot",

                  xaxis_title="Date",yaxis_title="Recovery Rate",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise_china.index, y=(datewise_china["Confirmed"]).diff().fillna(0),

                    mode='lines',name="China"))

fig.add_trace(go.Scatter(x=datewise_Italy.index, y=(datewise_Italy["Confirmed"]).diff().fillna(0),

                    mode='lines',name="Italy"))

fig.add_trace(go.Scatter(x=datewise_US.index, y=(datewise_US["Confirmed"]).diff().fillna(0),

                    mode='lines',name="United States"))

fig.add_trace(go.Scatter(x=datewise_Spain.index, y=(datewise_Spain["Confirmed"]).diff().fillna(0),

                    mode='lines',name="Spain"))

fig.add_trace(go.Scatter(x=datewise_Brazil.index, y=(datewise_Brazil["Confirmed"]).diff().fillna(0),

                    mode='lines',name="Brazil"))

fig.add_trace(go.Scatter(x=datewise_restofworld.index, y=(datewise_restofworld["Confirmed"]).diff().fillna(0),

                    mode='lines',name="Rest of the World"))

fig.update_layout(title="Daily increase in Number of Confirmed Cases",

                  xaxis_title="Date",yaxis_title="Number of Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise_china.index, y=(datewise_china["Deaths"]).diff().fillna(0),

                    mode='lines',name="China"))

fig.add_trace(go.Scatter(x=datewise_Italy.index, y=(datewise_Italy["Deaths"]).diff().fillna(0),

                    mode='lines',name="Italy"))

fig.add_trace(go.Scatter(x=datewise_US.index, y=(datewise_US["Deaths"]).diff().fillna(0),

                    mode='lines',name="United States"))

fig.add_trace(go.Scatter(x=datewise_Spain.index, y=(datewise_Spain["Deaths"]).diff().fillna(0),

                    mode='lines',name="Spain"))

fig.add_trace(go.Scatter(x=datewise_Brazil.index, y=(datewise_Brazil["Deaths"]).diff().fillna(0),

                    mode='lines',name="Brazil"))

fig.add_trace(go.Scatter(x=datewise_restofworld.index, y=(datewise_restofworld["Deaths"]).diff().fillna(0),

                    mode='lines',name="Rest of the World"))

fig.update_layout(title="Daily increase in Number of Death Cases",

                  xaxis_title="Date",yaxis_title="Number of Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
india_data=covid[covid["Country/Region"]=="India"]

datewise_india=india_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

print(datewise_india.iloc[-1])

print("Total Active Cases: ",datewise_india["Confirmed"].iloc[-1]-datewise_india["Recovered"].iloc[-1]-datewise_india["Deaths"].iloc[-1])

print("Total Closed Cases: ",datewise_india["Recovered"].iloc[-1]+datewise_india["Deaths"].iloc[-1])
fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise_india.index, y=datewise_india["Confirmed"],

                    mode='lines+markers',

                    name='Confirmed Cases'))

fig.add_trace(go.Scatter(x=datewise_india.index, y=datewise_india["Recovered"],

                    mode='lines+markers',

                    name='Recovered Cases'))

fig.add_trace(go.Scatter(x=datewise_india.index, y=datewise_india["Deaths"],

                    mode='lines+markers',

                    name='Death Cases'))

fig.update_layout(title="Growth of different types of cases in India",

                 xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=px.bar(x=datewise_india.index,y=datewise_india["Confirmed"]-datewise_india["Recovered"]-datewise_india["Deaths"])

fig.update_layout(title="Distribution of Number of Active Cases in India",

                  xaxis_title="Date",yaxis_title="Number of Cases",)

fig.show()
india_increase_confirm=[]

india_increase_recover=[]

india_increase_deaths=[]

for i in range(datewise_india.shape[0]-1):

    india_increase_confirm.append(((datewise_india["Confirmed"].iloc[i+1])/datewise_india["Confirmed"].iloc[i]))

    india_increase_recover.append(((datewise_india["Recovered"].iloc[i+1])/datewise_india["Recovered"].iloc[i]))

    india_increase_deaths.append(((datewise_india["Deaths"].iloc[i+1])/datewise_india["Deaths"].iloc[i]))

india_increase_confirm.insert(0,1)

india_increase_recover.insert(0,1)

india_increase_deaths.insert(0,1)



fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise_india.index, y=india_increase_confirm,

                    mode='lines',

                    name='Growth Factor of Confirmed Cases'))

fig.add_trace(go.Scatter(x=datewise_india.index, y=india_increase_recover,

                    mode='lines',

                    name='Growth Factor of Recovered Cases'))

fig.add_trace(go.Scatter(x=datewise_india.index, y=india_increase_deaths,

                    mode='lines',

                    name='Growth Factor of Death Cases'))

fig.update_layout(title="Datewise Growth Factor of Active and Closed cases in India",

                 xaxis_title="Date",yaxis_title="Growth Factor",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise_india.index, y=datewise_india["Confirmed"].diff().fillna(0),

                    mode='lines+markers',

                    name='Confirmed Cases'))

fig.add_trace(go.Scatter(x=datewise_india.index, y=datewise_india["Recovered"].diff().fillna(0),

                    mode='lines+markers',

                    name='Recovered Cases'))

fig.add_trace(go.Scatter(x=datewise_india.index, y=datewise_india["Deaths"].diff().fillna(0),

                    mode='lines+markers',

                    name='Death Cases'))

fig.update_layout(title="Daily increase in different types of cases in India",

                 xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
datewise_india["WeekOfYear"]=datewise_india.index.weekofyear



week_num_india=[]

india_weekwise_confirmed=[]

india_weekwise_recovered=[]

india_weekwise_deaths=[]

w=1

for i in list(datewise_india["WeekOfYear"].unique()):

    india_weekwise_confirmed.append(datewise_india[datewise_india["WeekOfYear"]==i]["Confirmed"].iloc[-1])

    india_weekwise_recovered.append(datewise_india[datewise_india["WeekOfYear"]==i]["Recovered"].iloc[-1])

    india_weekwise_deaths.append(datewise_india[datewise_india["WeekOfYear"]==i]["Deaths"].iloc[-1])

    week_num_india.append(w)

    w=w+1

    

fig=go.Figure()

fig.add_trace(go.Scatter(x=week_num_india, y=india_weekwise_confirmed,

                    mode='lines+markers',

                    name='Weekly Growth of Confirmed Cases'))

fig.add_trace(go.Scatter(x=week_num_india, y=india_weekwise_recovered,

                    mode='lines+markers',

                    name='Weekly Growth of Recovered Cases'))

fig.add_trace(go.Scatter(x=week_num_india, y=india_weekwise_deaths,

                    mode='lines+markers',

                    name='Weekly Growth of Death Cases'))

fig.update_layout(title="Weekly Growth of different types of Cases in India",

                 xaxis_title="Week Number",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(15,5))

sns.barplot(x=week_num_india,y=pd.Series(india_weekwise_confirmed).diff().fillna(0),ax=ax1)

sns.barplot(x=week_num_india,y=pd.Series(india_weekwise_deaths).diff().fillna(0),ax=ax2)

ax1.set_xlabel("Week Number")

ax2.set_xlabel("Week Number")

ax1.set_ylabel("Number of Confirmed Cases")

ax2.set_ylabel("Number of Death Cases")

ax1.set_title("India's Weekwise increase in Number of Confirmed Cases")

ax2.set_title("India's Weekwise increase in Number of Death Cases")
max_ind=datewise_india["Confirmed"].max()



print("It took",datewise_Italy[(datewise_Italy["Confirmed"]>0)&(datewise_Italy["Confirmed"]<=max_ind)].shape[0],"days in Italy to reach number of Confirmed Cases equivalent to India")

print("It took",datewise_US[(datewise_US["Confirmed"]>0)&(datewise_US["Confirmed"]<=max_ind)].shape[0],"days in USA to reach number of Confirmed Cases equivalent to India")

print("It took",datewise_Spain[(datewise_Spain["Confirmed"]>0)&(datewise_Spain["Confirmed"]<=max_ind)].shape[0],"days in Spain to reach number of Confirmed Cases equivalent to India")

print("It took",datewise_india[datewise_india["Confirmed"]>0].shape[0],"days in India to reach",max_ind,"Confirmed Cases")



fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise_Italy[(datewise_Italy["Confirmed"]>0)&(datewise_Italy["Confirmed"]<=max_ind)].index, y=datewise_Italy[(datewise_Italy["Confirmed"]>0)&(datewise_Italy["Confirmed"]<=max_ind)]["Confirmed"],

                    mode='lines',name="Italy"))

fig.add_trace(go.Scatter(x=datewise_US[(datewise_US["Confirmed"]>0)&(datewise_US["Confirmed"]<=max_ind)].index, y=datewise_US[(datewise_US["Confirmed"]>0)&(datewise_US["Confirmed"]<=max_ind)]["Confirmed"],

                    mode='lines',name="USA"))

fig.add_trace(go.Scatter(x=datewise_Spain[(datewise_Spain["Confirmed"]>0)&(datewise_Spain["Confirmed"]<=max_ind)].index, y=datewise_Spain[(datewise_Spain["Confirmed"]>0)&(datewise_Spain["Confirmed"]<=max_ind)]["Confirmed"],

                    mode='lines',name="Spain"))

fig.add_trace(go.Scatter(x=datewise_india.index, y=datewise_india["Confirmed"],

                    mode='lines',name="India"))

fig.update_layout(title="Growth of Recovered Cases with respect to India",

                 xaxis_title="Date",yaxis_title="Number of Confirmed Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise_Italy[(datewise_Italy["Confirmed"]>0)&(datewise_Italy["Confirmed"]<=max_ind)].index, 

                         y=datewise_Italy[(datewise_Italy["Confirmed"]>0)&(datewise_Italy["Confirmed"]<=max_ind)]["Confirmed"].diff().fillna(0),

                    mode='lines',name="Italy"))

fig.add_trace(go.Scatter(x=datewise_US[(datewise_US["Confirmed"]>0)&(datewise_US["Confirmed"]<=max_ind)].index, 

                         y=datewise_US[(datewise_US["Confirmed"]>0)&(datewise_US["Confirmed"]<=max_ind)]["Confirmed"].diff().fillna(0),

                    mode='lines',name="USA"))

fig.add_trace(go.Scatter(x=datewise_Spain[(datewise_Spain["Confirmed"]>0)&(datewise_Spain["Confirmed"]<=max_ind)].index,

                         y=datewise_Spain[(datewise_Spain["Confirmed"]>0)&(datewise_Spain["Confirmed"]<=max_ind)]["Confirmed"].diff().fillna(0),

                    mode='lines',name="Spain"))

fig.add_trace(go.Scatter(x=datewise_india.index, y=datewise_india["Confirmed"].diff().fillna(0),

                    mode='lines',name="India"))

fig.update_layout(title="Daily increase in Confirmed Cases",

                 xaxis_title="Date",yaxis_title="Number of Confirmed Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
datewise["Days Since"]=datewise.index-datewise.index[0]

datewise["Days Since"]=datewise["Days Since"].dt.days
train_ml=datewise.iloc[:int(datewise.shape[0]*0.95)]

valid_ml=datewise.iloc[int(datewise.shape[0]*0.95):]

model_scores=[]
lin_reg=LinearRegression(normalize=True)
lin_reg.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))
prediction_valid_linreg=lin_reg.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
model_scores.append(np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_linreg)))

print("Root Mean Square Error for Linear Regression: ",np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_linreg)))
plt.figure(figsize=(11,6))

prediction_linreg=lin_reg.predict(np.array(datewise["Days Since"]).reshape(-1,1))

linreg_output=[]

for i in range(prediction_linreg.shape[0]):

    linreg_output.append(prediction_linreg[i][0])



fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=datewise.index, y=linreg_output,

                    mode='lines',name="Linear Regression Best Fit Line",

                    line=dict(color='black', dash='dot')))

fig.update_layout(title="Confirmed Cases Linear Regression Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
train_ml=datewise.iloc[:int(datewise.shape[0]*0.95)]

valid_ml=datewise.iloc[int(datewise.shape[0]*0.95):]
poly = PolynomialFeatures(degree = 7) 
train_poly=poly.fit_transform(np.array(train_ml["Days Since"]).reshape(-1,1))

valid_poly=poly.fit_transform(np.array(valid_ml["Days Since"]).reshape(-1,1))

y=train_ml["Confirmed"]
linreg=LinearRegression(normalize=True)

linreg.fit(train_poly,y)
prediction_poly=linreg.predict(valid_poly)

rmse_poly=np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_poly))

model_scores.append(rmse_poly)

print("Root Mean Squared Error for Polynomial Regression: ",rmse_poly)
comp_data=poly.fit_transform(np.array(datewise["Days Since"]).reshape(-1,1))

plt.figure(figsize=(11,6))

predictions_poly=linreg.predict(comp_data)



fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=datewise.index, y=predictions_poly,

                    mode='lines',name="Polynomial Regression Best Fit",

                    line=dict(color='black', dash='dot')))

fig.update_layout(title="Confirmed Cases Polynomial Regression Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
new_prediction_poly=[]

for i in range(1,18):

    new_date_poly=poly.fit_transform(np.array(datewise["Days Since"].max()+i).reshape(-1,1))

    new_prediction_poly.append(linreg.predict(new_date_poly)[0])
train_ml=datewise.iloc[:int(datewise.shape[0]*0.95)]

valid_ml=datewise.iloc[int(datewise.shape[0]*0.95):]
#Intializing SVR Model

svm=SVR(C=1,degree=5,kernel='poly',epsilon=0.01)
#Fitting model on the training data

svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))
prediction_valid_svm=svm.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
model_scores.append(np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_svm)))

print("Root Mean Square Error for Support Vectore Machine: ",np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_svm)))
plt.figure(figsize=(11,6))

prediction_svm=svm.predict(np.array(datewise["Days Since"]).reshape(-1,1))

fig=go.Figure()

fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=datewise.index, y=prediction_svm,

                    mode='lines',name="Support Vector Machine Best fit Kernel",

                    line=dict(color='black', dash='dot')))

fig.update_layout(title="Confirmed Cases Support Vectore Machine Regressor Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
new_date=[]

new_prediction_lr=[]

new_prediction_svm=[]

for i in range(1,18):

    new_date.append(datewise.index[-1]+timedelta(days=i))

    new_prediction_lr.append(lin_reg.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0][0])

    new_prediction_svm.append(svm.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0])
pd.set_option('display.float_format', lambda x: '%.6f' % x)

model_predictions=pd.DataFrame(zip(new_date,new_prediction_lr,new_prediction_poly,new_prediction_svm),

                               columns=["Dates","Linear Regression Prediction","Polynonmial Regression Prediction","SVM Prediction"])

model_predictions.head()
model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]

valid=datewise.iloc[int(datewise.shape[0]*0.95):]

y_pred=valid.copy()
holt=Holt(np.asarray(model_train["Confirmed"])).fit(smoothing_level=1.2, smoothing_slope=0.8,optimized=False)
y_pred["Holt"]=holt.forecast(len(valid))

model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt"])))

print("Root Mean Square Error Holt's Linear Model: ",np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt"])))
fig=go.Figure()

fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=valid.index, y=valid["Confirmed"],

                    mode='lines+markers',name="Validation Data for Confirmed Cases",))

fig.add_trace(go.Scatter(x=valid.index, y=y_pred["Holt"],

                    mode='lines+markers',name="Prediction of Confirmed Cases",))

fig.update_layout(title="Confirmed Cases Holt's Linear Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
holt_new_date=[]

holt_new_prediction=[]

for i in range(1,18):

    holt_new_date.append(datewise.index[-1]+timedelta(days=i))

    holt_new_prediction.append(holt.forecast((len(valid)+i))[-1])



model_predictions["Holt's Linear Model Prediction"]=holt_new_prediction

model_predictions.head()
model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]

valid=datewise.iloc[int(datewise.shape[0]*0.95):]

y_pred=valid.copy()
es=ExponentialSmoothing(np.asarray(model_train['Confirmed']),seasonal_periods=3,trend='mul', seasonal='mul').fit()
y_pred["Holt's Winter Model"]=es.forecast(len(valid))
model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt's Winter Model"])))

print("Root Mean Square Error for Holt's Winter Model: ",np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt's Winter Model"])))
fig=go.Figure()

fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=valid.index, y=valid["Confirmed"],

                    mode='lines+markers',name="Validation Data for Confirmed Cases",))

fig.add_trace(go.Scatter(x=valid.index, y=y_pred["Holt\'s Winter Model"],

                    mode='lines+markers',name="Prediction of Confirmed Cases",))

fig.update_layout(title="Confirmed Cases Holt's Winter Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
holt_winter_new_prediction=[]

for i in range(1,18):

    holt_winter_new_prediction.append(es.forecast((len(valid)+i))[-1])

model_predictions["Holt's Winter Model Prediction"]=holt_winter_new_prediction

model_predictions.head()
model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]

valid=datewise.iloc[int(datewise.shape[0]*0.95):]

y_pred=valid.copy()
model_ar= auto_arima(model_train["Confirmed"],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=2,max_q=0,

                   suppress_warnings=True,stepwise=False,seasonal=False)

model_ar.fit(model_train["Confirmed"])
prediction_ar=model_ar.predict(len(valid))

y_pred["AR Model Prediction"]=prediction_ar
model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["AR Model Prediction"])))

print("Root Mean Square Error for AR Model: ",np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["AR Model Prediction"])))
fig=go.Figure()

fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=valid.index, y=valid["Confirmed"],

                    mode='lines+markers',name="Validation Data for Confirmed Cases",))

fig.add_trace(go.Scatter(x=valid.index, y=y_pred["AR Model Prediction"],

                    mode='lines+markers',name="Prediction of Confirmed Cases",))

fig.update_layout(title="Confirmed Cases AR Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
AR_model_new_prediction=[]

for i in range(1,18):

    AR_model_new_prediction.append(model_ar.predict(len(valid)+i)[-1])

model_predictions["AR Model Prediction"]=AR_model_new_prediction

model_predictions.head()
model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]

valid=datewise.iloc[int(datewise.shape[0]*0.95):]

y_pred=valid.copy()
model_ma= auto_arima(model_train["Confirmed"],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=0,max_q=4,

                   suppress_warnings=True,stepwise=False,seasonal=False)

model_ma.fit(model_train["Confirmed"])
prediction_ma=model_ma.predict(len(valid))

y_pred["MA Model Prediction"]=prediction_ma
model_scores.append(np.sqrt(mean_squared_error(valid["Confirmed"],prediction_ma)))

print("Root Mean Square Error for MA Model: ",np.sqrt(mean_squared_error(valid["Confirmed"],prediction_ma)))
fig=go.Figure()

fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=valid.index, y=valid["Confirmed"],

                    mode='lines+markers',name="Validation Data for Confirmed Cases",))

fig.add_trace(go.Scatter(x=valid.index, y=y_pred["MA Model Prediction"],

                    mode='lines+markers',name="Prediction for Confirmed Cases",))

fig.update_layout(title="Confirmed Cases MA Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
MA_model_new_prediction=[]

for i in range(1,18):

    MA_model_new_prediction.append(model_ma.predict(len(valid)+i)[-1])

model_predictions["MA Model Prediction"]=MA_model_new_prediction

model_predictions.head()
model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]

valid=datewise.iloc[int(datewise.shape[0]*0.95):]

y_pred=valid.copy()
model_arima= auto_arima(model_train["Confirmed"],trace=True, error_action='ignore', start_p=1,start_q=1,max_p=2,max_q=2,

                   suppress_warnings=True,stepwise=False,seasonal=False)

model_arima.fit(model_train["Confirmed"])
prediction_arima=model_arima.predict(len(valid))

y_pred["ARIMA Model Prediction"]=prediction_arima
model_scores.append(np.sqrt(mean_squared_error(valid["Confirmed"],prediction_arima)))

print("Root Mean Square Error for ARIMA Model: ",np.sqrt(mean_squared_error(valid["Confirmed"],prediction_arima)))
fig=go.Figure()

fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=valid.index, y=valid["Confirmed"],

                    mode='lines+markers',name="Validation Data for Confirmed Cases",))

fig.add_trace(go.Scatter(x=valid.index, y=y_pred["ARIMA Model Prediction"],

                    mode='lines+markers',name="Prediction for Confirmed Cases",))

fig.update_layout(title="Confirmed Cases ARIMA Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
ARIMA_model_new_prediction=[]

for i in range(1,18):

    ARIMA_model_new_prediction.append(model_arima.predict(len(valid)+i)[-1])

model_predictions["ARIMA Model Prediction"]=ARIMA_model_new_prediction

model_predictions.head()
model_sarima= auto_arima(model_train["Confirmed"],trace=True, error_action='ignore', 

                         start_p=0,start_q=0,max_p=2,max_q=2,m=30,

                   suppress_warnings=True,stepwise=True,seasonal=True)

model_sarima.fit(model_train["Confirmed"])
prediction_sarima=model_sarima.predict(len(valid))

y_pred["SARIMA Model Prediction"]=prediction_sarima
model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["SARIMA Model Prediction"])))

print("Root Mean Square Error for SARIMA Model: ",np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["SARIMA Model Prediction"])))
fig=go.Figure()

fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=valid.index, y=valid["Confirmed"],

                    mode='lines+markers',name="Validation Data for Confirmed Cases",))

fig.add_trace(go.Scatter(x=valid.index, y=y_pred["SARIMA Model Prediction"],

                    mode='lines+markers',name="Prediction for Confirmed Cases",))

fig.update_layout(title="Confirmed Cases SARIMA Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
SARIMA_model_new_prediction=[]

for i in range(1,18):

    SARIMA_model_new_prediction.append(model_sarima.predict(len(valid)+i)[-1])

model_predictions["SARIMA Model Prediction"]=SARIMA_model_new_prediction

model_predictions.head()
prophet_c=Prophet(interval_width=0.95,weekly_seasonality=True,)

prophet_confirmed=pd.DataFrame(zip(list(datewise.index),list(datewise["Confirmed"])),columns=['ds','y'])
prophet_c.fit(prophet_confirmed)
forecast_c=prophet_c.make_future_dataframe(periods=17)

forecast_confirmed=forecast_c.copy()
confirmed_forecast=prophet_c.predict(forecast_c)

#print(confirmed_forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']])
model_scores.append(np.sqrt(mean_squared_error(datewise["Confirmed"],confirmed_forecast['yhat'].head(datewise.shape[0]))))

print("Root Mean Squared Error for Prophet Model: ",np.sqrt(mean_squared_error(datewise["Confirmed"],confirmed_forecast['yhat'].head(datewise.shape[0]))))
print(prophet_c.plot(confirmed_forecast))
print(prophet_c.plot_components(confirmed_forecast))
model_names=["Linear Regression","Polynomial Regression","Support Vector Machine Regressor","Holt's Linear","Holt's Winter Model",

            "Auto Regressive Model (AR)","Moving Average Model (MA)","ARIMA Model","SARIMA Model","Facebook's Prophet Model"]

model_summary=pd.DataFrame(zip(model_names,model_scores),columns=["Model Name","Root Mean Squared Error"]).sort_values(["Root Mean Squared Error"])

model_summary
model_predictions["Prophet's Prediction"]=list(confirmed_forecast["yhat"].tail(17))

model_predictions["Prophet's Upper Bound"]=list(confirmed_forecast["yhat_upper"].tail(17))

model_predictions.head()
fig=go.Figure()

fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Deaths"],

                    mode='lines+markers',name="Death Cases"))

fig.update_layout(title="Death Cases",

                 xaxis_title="Date",yaxis_title="Number of Death Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]

valid=datewise.iloc[int(datewise.shape[0]*0.95):]

y_pred=valid.copy()
model_arima_deaths=auto_arima(model_train["Deaths"],trace=True, error_action='ignore', start_p=0,start_q=0,

                              max_p=3,max_q=3,suppress_warnings=True,stepwise=False,seasonal=False)     

model_arima_deaths.fit(model_train["Deaths"])
predictions_deaths=model_arima_deaths.predict(len(valid))

y_pred["ARIMA Death Prediction"]=predictions_deaths
print("Root Mean Square Error: ",np.sqrt(mean_squared_error(valid["Deaths"],predictions_deaths)))
fig=go.Figure()

fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Deaths"],

                    mode='lines+markers',name="Train Data for Death Cases"))

fig.add_trace(go.Scatter(x=valid.index, y=valid["Deaths"],

                    mode='lines+markers',name="Validation Data for Death Cases",))

fig.add_trace(go.Scatter(x=valid.index, y=y_pred["ARIMA Death Prediction"],

                    mode='lines+markers',name="Prediction for Death Cases",))

fig.update_layout(title="Death Cases ARIMA Model Prediction",

                 xaxis_title="Date",yaxis_title="Death Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
ARIMA_model_death_forecast=[]

for i in range(1,18):

    ARIMA_model_death_forecast.append(model_arima_deaths.predict(len(valid)+i)[-1])
pd.DataFrame(zip(new_date,ARIMA_model_death_forecast),columns=["Deaths","ARIMA Model Death Forecast"]).head()