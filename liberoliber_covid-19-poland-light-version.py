import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import datetime as dt

from datetime import timedelta

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error,r2_score

import statsmodels.api as sm

from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing

from fbprophet import Prophet

from sklearn.preprocessing import PolynomialFeatures

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.ar_model import AR

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

#pd.set_option('display.float_format', lambda x: '%.6f' % x)

import datetime
COUNTRY = "Poland"





df    = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

covid = df



df['Country/Region'] = df['Country/Region'].astype('category')

poland = df[df.loc[:, 'Country/Region'] == COUNTRY]

covid.head()
#Converting "Observation Date" into Datetime format

covid["ObservationDate"] =pd.to_datetime(covid["ObservationDate"])

covid["Last Update"]     =pd.to_datetime(covid["Last Update"])



covid.head()
#polonia = pd.concat([poland_only_df.set_index('Last Update'),covid.set_index('Last Update')])

#polonia.rename(columns={'Last Update': 'Date','Voivodeship': 'Province/State'}, inplace=True)

#polonia
#Dropping column as SNo is of no use, and "Province/State" contains too many missing values

covid.drop(["SNo"],1,inplace=True)
#Grouping different types of cases as per the date

datewise=covid.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise
from datetime import date



x = date.today()



x.strftime("%b %d %Y")

today = x.strftime("%d-%m-%Y")

#today


print("General Information about the spread across the world on " + str(today) +".")



print(" ")





print("Total number of countries with Disease Spread:         ",len(covid["Country/Region"].unique()))

print("Total number of Confirmed Cases around the World:   {:.0f} ".format(datewise["Confirmed"].iloc[-1]))

print("Total number of Recovered Cases around the World:   {:.0f}".format(datewise["Recovered"].iloc[-1]))

print("Total number of Deaths Cases around the World:       {:.0f}".format(datewise["Deaths"].iloc[-1]))

print("Total number of Active Cases around the World:     ",int((datewise["Confirmed"].iloc[-1]-datewise["Recovered"].iloc[-1]-datewise["Deaths"].iloc[-1])))

print("Total number of Closed Cases around the World:     ",int(datewise["Recovered"].iloc[-1]+datewise["Deaths"].iloc[-1]))

print(" ")

print("Number of Confirmed Cases per Day around the World:  ",int(np.round(datewise["Confirmed"].iloc[-1]/datewise.shape[0])))

print("Number of Recovered Cases per Day around the World:  ",int(np.round(datewise["Recovered"].iloc[-1]/datewise.shape[0])))

print("Number of Death Cases per Day around the World:       ",int(np.round(datewise["Deaths"].iloc[-1]/datewise.shape[0])))

print("Number of Confirmed Cases per hour around the World:  ",int(np.round(datewise["Confirmed"].iloc[-1]/((datewise.shape[0])*24))))

print("Number of Recovered Cases per hour around the World:   ",int(np.round(datewise["Recovered"].iloc[-1]/((datewise.shape[0])*24))))

print("Number of Death Cases per hour around the World:       ",int(np.round(datewise["Deaths"].iloc[-1]/((datewise.shape[0])*24))))



print(" ")
plt.figure(figsize=(25,8))

sns.barplot(x=datewise.index.date, y=datewise["Confirmed"]-datewise["Recovered"]-datewise["Deaths"])

plt.title("Distribution Plot for Active Cases Cases over Date")

plt.xticks(rotation=90)



plt.savefig('001pl.png')
plt.figure(figsize=(25,8))

sns.barplot(x=datewise.index.date, y=datewise["Recovered"]+datewise["Deaths"])

plt.title("Distribution Plot for Closed Cases Cases over Date")

plt.xticks(rotation=90)

plt.savefig('002pl.png')
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



#plt.figure(figsize=(8,5))

#plt.plot(week_num,weekwise_confirmed,linewidth=3)

#plt.plot(week_num,weekwise_recovered,linewidth=3)

#plt.plot(week_num,weekwise_deaths,linewidth=3)

#plt.ylabel("Number of Cases")

#plt.xlabel("Week Number")

#plt.title("Weekly progress of Different Types of Cases")

#plt.xlabel
fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(25,8))

sns.barplot(x=week_num,y=pd.Series(weekwise_confirmed).diff().fillna(0),ax=ax1)

sns.barplot(x=week_num,y=pd.Series(weekwise_deaths).diff().fillna(0),ax=ax2)

ax1.set_xlabel("Week Number")

ax2.set_xlabel("Week Number")

ax1.set_ylabel("Number of Confirmed Cases")

ax2.set_ylabel("Number of Death Cases")

ax1.set_title("Weekly increase in Number of Confirmed Cases")

ax2.set_title("Weekly increase in Number of Death Cases")

plt.savefig('003pl.png')
#plt.figure(figsize=(25,8))

#plt.plot(datewise["Confirmed"],marker="o",label="Confirmed Cases")

#plt.plot(datewise["Recovered"],marker="*",label="Recovered Cases")

#plt.plot(datewise["Deaths"],marker="^",label="Death Cases")

#plt.ylabel("Number of Patients")

#plt.xlabel("Dates")

#plt.xticks(rotation=90)

#plt.title("Growth of different Types of Cases over Time")

#plt.legend()

#plt.savefig('004pl.png')
daily_increase_confirm=[]

daily_increase_recovered=[]

daily_increase_deaths=[]

for i in range(datewise.shape[0]-1):

    daily_increase_confirm.append(((datewise["Confirmed"].iloc[i+1]/datewise["Confirmed"].iloc[i])))

    daily_increase_recovered.append(((datewise["Recovered"].iloc[i+1]/datewise["Recovered"].iloc[i])))

    daily_increase_deaths.append(((datewise["Deaths"].iloc[i+1]/datewise["Deaths"].iloc[i])))

daily_increase_confirm.insert(0,1)

daily_increase_recovered.insert(0,1)

daily_increase_deaths.insert(0,1)
#Calculating countrywise Moratality and Recovery Rate

countrywise=covid[covid["ObservationDate"]==covid["ObservationDate"].max()].groupby(["Country/Region"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'}).sort_values(["Confirmed"],ascending=False)

countrywise["Mortality"]=(countrywise["Deaths"]/countrywise["Confirmed"])*100

countrywise["Recovery"]=(countrywise["Recovered"]/countrywise["Confirmed"])*100
china_data=covid[covid["Country/Region"]=="Mainland China"]

Italy_data=covid[covid["Country/Region"]=="Italy"]

US_data=covid[covid["Country/Region"]=="US"]

poland_data=covid[covid["Country/Region"]=="Poland"]

brazil_data=covid[covid["Country/Region"]=="Brazil"]

rest_of_world=covid[(covid["Country/Region"]!="Mainland China")&(covid["Country/Region"]!="Italy")&(covid["Country/Region"]!="US")&(covid["Country/Region"]!="Spain")]



datewise_china=china_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise_Italy=Italy_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise_poland=poland_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise_US=US_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise_brazil=brazil_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise_restofworld=rest_of_world.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(25,8))

ax1.plot(datewise_poland["Confirmed"],label="Confirmed Cases of Poland",linewidth=3)

ax1.plot(datewise_brazil["Confirmed"],label="Confirmed Cases of Brazil",linewidth=3)

ax1.plot(datewise_US["Confirmed"],label="Confirmed Cases of USA",linewidth=3)

#ax1.plot(datewise_Spain["Confirmed"],label="Confirmed Cases of Spain",linewidth=3)

#ax1.plot(datewise_restofworld["Confirmed"],label="Confirmed Cases of Rest of the World",linewidth=3)

ax1.set_title("Confirmed Cases Plot")

ax1.set_ylabel("Number of Patients")

ax1.set_xlabel("Dates")

ax1.legend()

for tick in ax1.get_xticklabels():

    tick.set_rotation(90)

ax2.plot(datewise_poland["Recovered"],label="Recovered Cases of Poland",linewidth=3)

ax2.plot(datewise_brazil["Recovered"],label="Recovered Cases of Brazil",linewidth=3)

ax2.plot(datewise_US["Recovered"],label="Recovered Cases of US",linewidth=3)

#ax2.plot(datewise_Spain["Recovered"],label="Recovered Cases Spain",linewidth=3)

#ax2.plot(datewise_restofworld["Recovered"],label="Recovered Cases of Rest of the World",linewidth=3)

ax2.set_title("Recovered Cases Plot")

ax2.set_ylabel("Number of Patients")

ax2.set_xlabel("Dates")

ax2.legend()

for tick in ax2.get_xticklabels():

    tick.set_rotation(90)

ax3.plot(datewise_poland["Deaths"],label='Death Cases of Poland',linewidth=3)

ax3.plot(datewise_brazil["Deaths"],label='Death Cases of Brazil',linewidth=3)

ax3.plot(datewise_US["Deaths"],label='Death Cases of USA',linewidth=3)

#ax3.plot(datewise_Spain["Deaths"],label='Death Cases Spain',linewidth=3)

#ax3.plot(datewise_restofworld["Deaths"],label="Deaths Cases of Rest of the World",linewidth=3)

ax3.set_title("Death Cases Plot")

ax3.set_ylabel("Number of Patients")

ax3.set_xlabel("Dates")

ax3.legend()

for tick in ax3.get_xticklabels():

    tick.set_rotation(90)

    

plt.savefig('013pl.png')
country = "Poland"

poland_data=covid[covid["Country/Region"]=="Poland"]



poland_data=covid[covid["Country/Region"]==country]

datewise_poland=poland_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

print(datewise_poland.iloc[-1])

print("Total Active Cases: ",datewise_poland["Confirmed"].iloc[-1]-datewise_poland["Recovered"].iloc[-1]-datewise_poland["Deaths"].iloc[-1])

print("Total Closed Cases: ",datewise_poland["Recovered"].iloc[-1]+datewise_poland["Deaths"].iloc[-1])
print("COVID-19 situation in " +country +" on " + str(today) +" in numbers.")



print(" ")



print("Total number of Confirmed Cases:                {:.0f} ".format(datewise_poland["Confirmed"].iloc[-1]))

print("Total number of Recovered Cases:                 {:.0f}".format(datewise_poland["Recovered"].iloc[-1]))

print("Total number of Deaths Cases:                     {:.0f}".format(datewise_poland["Deaths"].iloc[-1]))

print("Total number of Active Cases:                  ",int((datewise_poland["Confirmed"].iloc[-1]-datewise_poland["Recovered"].iloc[-1]-datewise_poland["Deaths"].iloc[-1])))

print("Total number of Closed Cases:                   ",int(datewise_poland["Recovered"].iloc[-1]+datewise_poland["Deaths"].iloc[-1]))

print(" ")

print("Number of Confirmed Cases per Day:               ",int(np.round(datewise_poland["Confirmed"].iloc[-1]/datewise_poland.shape[0])))

print("Number of Recovered Cases per Day:               ",int(np.round(datewise_poland["Recovered"].iloc[-1]/datewise_poland.shape[0])))

print("Number of Death Cases per Day:                    ",int(np.round(datewise_poland["Deaths"].iloc[-1]/datewise_poland.shape[0])))

print("Number of Confirmed Cases per hour:               ",int(np.round(datewise_poland["Confirmed"].iloc[-1]/((datewise.shape[0])*24))))

print("Number of Recovered Cases per hour:                ",int(np.round(datewise_poland["Recovered"].iloc[-1]/((datewise_poland.shape[0])*24))))

#print("Number of Death Cases per hour:                ",int(np.round(datewise_poland["Deaths"].iloc[-1]/((datewise_poland.shape[0])*24))))



print(" ")
#Calculating the Mortality Rate and Recovery Rate Worldwide

datewise["Mortality Rate"]=(datewise["Deaths"]/datewise["Confirmed"])*100

datewise["Recovery Rate"]=(datewise["Recovered"]/datewise["Confirmed"])*100

datewise["Active Cases"]=datewise["Confirmed"]-datewise["Recovered"]-datewise["Deaths"]

datewise["Closed Cases"]=datewise["Recovered"]+datewise["Deaths"]



#Calculating the Mortality Rate and Recovery Rate local

datewise_poland["Mortality Rate"]=(datewise_poland["Deaths"]/datewise_poland["Confirmed"])*100

datewise_poland["Recovery Rate"]=(datewise_poland["Recovered"]/datewise_poland["Confirmed"])*100

datewise_poland["Active Cases"]=datewise_poland["Confirmed"]-datewise_poland["Recovered"]-datewise_poland["Deaths"]

datewise_poland["Closed Cases"]=datewise_poland["Recovered"]+datewise_poland["Deaths"]



#Plotting Mortality and Recovery Rate 

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(25,8))

ax1.plot(datewise_poland["Mortality Rate"],label='Mortality Rate',linewidth=3)

ax1.axhline(datewise_poland["Mortality Rate"].mean(),linestyle='--',color='black',label="Mean Mortality Rate")

ax1.set_ylabel("Mortality Rate in " + country +".")

ax1.set_xlabel("Timestamp")

ax1.set_title("Overall Datewise Mortality Rate in " + country +".")

ax1.legend()



for tick in ax1.get_xticklabels():

    tick.set_rotation(90)

ax2.plot(datewise_poland["Recovery Rate"],label="Recovery Rate",linewidth=3)

ax2.axhline(datewise_poland["Recovery Rate"].mean(),linestyle='--',color='black',label="Mean Recovery Rate")

ax2.set_ylabel("Recovery Rate in " + country +".")

ax2.set_xlabel("Timestamp")

ax2.set_title("Overall Datewise Recovery Rate in " + country +".")

ax2.legend()



for tick in ax2.get_xticklabels():

    tick.set_rotation(90)

    

plt.savefig('014pl.png')



precision = 2



print("Average Mortality and Recovery Rates in " + country + ".(Between parenthesis the worldwide rates).")

print()



print( "Average Mortality Rate: {:.{}f}".format( datewise_poland["Mortality Rate"].mean(), precision ) + " ({:.{}f}".format( datewise["Mortality Rate"].mean(), precision )+")") 



print( "Average Recovery Rate: {:.{}f}".format( datewise_poland["Recovery Rate"].mean(), precision ) + " ({:.{}f}".format( datewise["Recovery Rate"].mean(), precision )+")")
datewise_poland["WeekOfYear"]=datewise_poland.index.weekofyear



week_num_poland=[]

poland_weekwise_confirmed=[]

poland_weekwise_recovered=[]

poland_weekwise_deaths=[]

w=1

for i in list(datewise_poland["WeekOfYear"].unique()):

    poland_weekwise_confirmed.append(datewise_poland[datewise_poland["WeekOfYear"]==i]["Confirmed"].iloc[-1])

    poland_weekwise_recovered.append(datewise_poland[datewise_poland["WeekOfYear"]==i]["Recovered"].iloc[-1])

    poland_weekwise_deaths.append(datewise_poland[datewise_poland["WeekOfYear"]==i]["Deaths"].iloc[-1])

    week_num_poland.append(w)

    w=w+1
fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(25,8))

sns.barplot(x=week_num_poland,y=pd.Series(poland_weekwise_confirmed).diff().fillna(0),ax=ax1)

sns.barplot(x=week_num_poland,y=pd.Series(poland_weekwise_deaths).diff().fillna(0),ax=ax2)

ax1.set_xlabel("Week Number")

ax2.set_xlabel("Week Number")

ax1.set_ylabel("Number of Confirmed Cases")

ax2.set_ylabel("Number of Death Cases")

ax1.set_title("Weekwise increase in Number of Confirmed Cases in " + country +".")

ax2.set_title("Weekwise increase in Number of Death Cases in " + country +".")



plt.savefig('021pl.png')
confirmed_covid19 = int(countrywise["Confirmed"].sum())

confirmed_covid19
deaths_covid19 = int(countrywise["Deaths"].sum())

deaths_covid19
epidemics = pd.DataFrame({

    'epidemic'   : ['COVID-19', 'SARS', 'EBOLA', 'MERS', 'H1N1'],

    'start year' : [2019, 2003, 2014, 2012, 2009],

    'end year'   : [2020, 2004, 2016, 2017, 2010],

    'confirmed'  : [int(confirmed_covid19), 8096, 28646, 2494, 6724149],

    'deaths'     : [deaths_covid19, 774, 11323, 858, 19654]

})



epidemics['mortality'] = round((epidemics['deaths']/epidemics['confirmed'])*100, 2)



epidemics.head()
import plotly.express as px



temp = epidemics.melt(id_vars='epidemic', value_vars=['confirmed', 'deaths', 'mortality'],

                      var_name='Case', value_name='Value')



fig = px.bar(temp, x="epidemic", y="Value", color='epidemic', text='Value', facet_col="Case",

             color_discrete_sequence = px.colors.qualitative.Bold)

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.update_yaxes(showticklabels=False)

fig.layout.yaxis2.update(matches=None)

fig.layout.yaxis3.update(matches=None)





fig.show()

#fig.write_image("images/031pl_pandemia.png")
datewise = datewise_poland

datewise["Days Since"]=datewise.index-datewise.index[0]

datewise["Days Since"]=datewise["Days Since"].dt.days
prophet_c=Prophet(interval_width=0.95,weekly_seasonality=True,)

prophet_confirmed=pd.DataFrame(zip(list(datewise.index),list(datewise["Confirmed"])),columns=['ds','y'])
prophet_c.fit(prophet_confirmed)
forecast_c=prophet_c.make_future_dataframe(periods=16)

forecast_confirmed=forecast_c.copy()
confirmed_forecast=prophet_c.predict(forecast_c)

#print(confirmed_forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']])
print(prophet_c.plot(confirmed_forecast))
print(prophet_c.plot_components(confirmed_forecast))
model_names=["Facebook's Prophet Model"]

pd.DataFrame(zip(model_names,model_scores),columns=["Model Name","Root Mean Squared Error"]).sort_values(["Root Mean Squared Error"])

print(datewise_poland.iloc[-1])
model_predictions["Prophet's Prediction"]=list(confirmed_forecast["yhat"].tail(17))
df = pd.DataFrame(model_predictions,columns=["Dates","Prophet's Prediction"])



df['Dates'] = pd.to_datetime(df['Dates'])

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('precision', 0)



#df = df[(df['yhat']>0)]

df.rename(columns={'Prophet\'s Prediction': 'Prophet'}, inplace=True)

forecast_table = df



df
forecast_summary = pd.concat([cases_forecast.set_index('Dates'), deaths_forecast.set_index('Dates')], axis=1, join='inner')

forecast_summary.rename(columns={'Dates': 'Date','Forecast': 'Cases','Deaths Forecast': 'Deaths'}, inplace=True)
plt.figure(figsize=(25,8))

plt.plot(datewise_poland["Confirmed"],label="Actual Cases")

plt.bar(df.Dates, forecast_summary.Cases, color='royalblue', alpha=0.7)





plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.legend(['Confirmed Cases until '+ str(today)])



plt.savefig('022pl.png')

plt.show()
datewise = datewise_poland

datewise["Days Since"]=datewise.index-datewise.index[0]

datewise["Days Since"]=datewise["Days Since"].dt.days
prophet_c=Prophet(interval_width=0.95,weekly_seasonality=True,)

prophet_Deaths=pd.DataFrame(zip(list(datewise.index),list(datewise["Deaths"])),columns=['ds','y'])
prophet_c.fit(prophet_Deaths)
forecast_c=prophet_c.make_future_dataframe(periods=15)

forecast_Deaths=forecast_c.copy()
Deaths_forecast=prophet_c.predict(forecast_c)

#print(confirmed_forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']])
model_scores.append(np.sqrt(mean_squared_error(datewise["Deaths"],Deaths_forecast['yhat'].head(datewise.shape[0]))))



precision = 2

#print( "{:.{}f}".format( pi, precision )) 



print( "Root Mean Squared Error for Prophet Model: {:.{}f}".format( np.sqrt(mean_squared_error(datewise["Deaths"],Deaths_forecast['yhat'].head(datewise.shape[0]))), precision ))
print(prophet_c.plot(Deaths_forecast))
print(prophet_c.plot_components(Deaths_forecast))
model_names=["Facebook's Prophet Model"]

pd.DataFrame(zip(model_names,model_scores),columns=["Model Name","Root Mean Squared Error"]).sort_values(["Root Mean Squared Error"])

print(datewise_poland.iloc[-1])
model_predictions["Prophet's Prediction"]=list(Deaths_forecast["yhat"].tail(17))



#model_predictions.head()
df = pd.DataFrame(model_predictions,columns=["Dates","Prophet's Prediction"])



df['Dates'] = pd.to_datetime(df['Dates'])

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('precision', 0)



#df = df[(df['yhat']>0)]

df.rename(columns={'Prophet\'s Prediction': 'Prophet'}, inplace=True)

forecast_table = df
df = pd.DataFrame(model_predictions,columns=["Dates","Prophet's Prediction"])



df['Dates'] = pd.to_datetime(df['Dates'])

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('precision', 0)



#df = df[(df['yhat']>0)]

df.rename(columns={'Prophet\'s Prediction': 'Deaths Forecast'}, inplace=True)

forecast_table = df

deaths_forecast = df

df
plt.figure(figsize=(25,8))

plt.plot(datewise_poland["Deaths"],label="Deaths")

plt.bar(df.Dates, forecast_summary.Deaths, color='red', alpha=0.7)





plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.legend(['Deaths until '+ str(today)])



plt.show()