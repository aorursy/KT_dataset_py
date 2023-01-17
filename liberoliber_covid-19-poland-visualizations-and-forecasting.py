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
# install Required Libraries

# ==============



! pip install calmap

! pip install plotly psutil requests

!yes Y | conda install -c plotly plotly-orca
COUNTRY = "Poland"



population_df  = pd.read_csv("../input/population-by-country-2020/population_by_country_2020.csv")

poland_only_df = pd.read_csv("../input/covid19-in-poland-dataset/2020-04-28.csv")

df             = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

covid          = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")



df['Country/Region'] = df['Country/Region'].astype('category')

poland = df[df.loc[:, 'Country/Region'] == COUNTRY]

covid.head()
population_df.head()
#Converting "Observation Date" into Datetime format

covid["ObservationDate"]=pd.to_datetime(covid["ObservationDate"])

covid["Last Update"]=pd.to_datetime(covid["Last Update"])

covid.head()
#Converting "Observation Date" into Datetime format



poland_only_df["Last Update"]=pd.to_datetime(poland_only_df["Last Update"])

poland_only_df.head()
polonia = pd.concat([poland_only_df.set_index('Last Update'),covid.set_index('Last Update')])

polonia.rename(columns={'Last Update': 'Date','Voivodeship': 'Province/State'}, inplace=True)

polonia
#covid = polonia
#Dropping column as SNo is of no use, and "Province/State" contains too many missing values

covid.drop(["SNo"],1,inplace=True)
#Converting "Observation Date" into Datetime format

covid["ObservationDate"]=pd.to_datetime(covid["ObservationDate"])
#Grouping different types of cases as per the date

datewise=covid.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
from datetime import date



x = date.today()



x.strftime("%b %d %Y")

today = x.strftime("%d-%m-%Y")

#today


print("General Information about the spread across the world on " + str(today) +".")



print(" ")





print("Total number of countries with Disease Spread:      ",len(covid["Country/Region"].unique()))

print("Total number of Confirmed Cases around the World:  {:.0f} ".format(datewise["Confirmed"].iloc[-1]))

print("Total number of Recovered Cases around the World:   {:.0f}".format(datewise["Recovered"].iloc[-1]))

print("Total number of Deaths Cases around the World:       {:.0f}".format(datewise["Deaths"].iloc[-1]))

print("Total number of Active Cases around the World:     ",int((datewise["Confirmed"].iloc[-1]-datewise["Recovered"].iloc[-1]-datewise["Deaths"].iloc[-1])))

print("Total number of Closed Cases around the World:     ",int(datewise["Recovered"].iloc[-1]+datewise["Deaths"].iloc[-1]))

print("Number of Confirmed Cases per Day around the World: ",int(np.round(datewise["Confirmed"].iloc[-1]/datewise.shape[0])))

print("Number of Recovered Cases per Day around the World:  ",int(np.round(datewise["Recovered"].iloc[-1]/datewise.shape[0])))

print("Number of Death Cases per Day around the World:      ",int(np.round(datewise["Deaths"].iloc[-1]/datewise.shape[0])))

print("Number of Confirmed Cases per hour around the World:  ",int(np.round(datewise["Confirmed"].iloc[-1]/((datewise.shape[0])*24))))

print("Number of Recovered Cases per hour around the World:  ",int(np.round(datewise["Recovered"].iloc[-1]/((datewise.shape[0])*24))))

print("Number of Death Cases per hour around the World:      ",int(np.round(datewise["Deaths"].iloc[-1]/((datewise.shape[0])*24))))



print(" ")

print("Acknowledgements:")

print("Thanks to the WHO and Johns Hopkins University for making the ")

print("data available for educational and academic research purposes - Jair Ribeiro")
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



plt.figure(figsize=(8,5))

plt.plot(week_num,weekwise_confirmed,linewidth=3)

plt.plot(week_num,weekwise_recovered,linewidth=3)

plt.plot(week_num,weekwise_deaths,linewidth=3)

plt.ylabel("Number of Cases")

plt.xlabel("Week Number")

plt.title("Weekly progress of Different Types of Cases")

plt.xlabel
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
plt.figure(figsize=(25,8))

plt.plot(datewise["Confirmed"],marker="o",label="Confirmed Cases")

plt.plot(datewise["Recovered"],marker="*",label="Recovered Cases")

plt.plot(datewise["Deaths"],marker="^",label="Death Cases")

plt.ylabel("Number of Patients")

plt.xlabel("Dates")

plt.xticks(rotation=90)

plt.title("Growth of different Types of Cases over Time")

plt.legend()

plt.savefig('004pl.png')
#Calculating the Mortality Rate and Recovery Rate

datewise["Mortality Rate"]=(datewise["Deaths"]/datewise["Confirmed"])*100

datewise["Recovery Rate"]=(datewise["Recovered"]/datewise["Confirmed"])*100

datewise["Active Cases"]=datewise["Confirmed"]-datewise["Recovered"]-datewise["Deaths"]

datewise["Closed Cases"]=datewise["Recovered"]+datewise["Deaths"]



#Plotting Mortality and Recovery Rate 

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(25,8))

ax1.plot(datewise["Mortality Rate"],label='Mortality Rate',linewidth=3)

ax1.axhline(datewise["Mortality Rate"].mean(),linestyle='--',color='black',label="Mean Mortality Rate")

ax1.set_ylabel("Mortality Rate")

ax1.set_xlabel("Timestamp")

ax1.set_title("Overall Datewise Mortality Rate")

ax1.legend()

for tick in ax1.get_xticklabels():

    tick.set_rotation(90)

ax2.plot(datewise["Recovery Rate"],label="Recovery Rate",linewidth=3)

ax2.axhline(datewise["Recovery Rate"].mean(),linestyle='--',color='black',label="Mean Recovery Rate")

ax2.set_ylabel("Recovery Rate")

ax2.set_xlabel("Timestamp")

ax2.set_title("Overall Datewise Recovery Rate")

ax2.legend()

for tick in ax2.get_xticklabels():

    tick.set_rotation(90)

    



precision = 2

#print( "{:.{}f}".format( pi, precision )) 



print( "Average Mortality Rate: {:.{}f}".format( datewise["Mortality Rate"].mean(), precision )) 

#print( "Median Mortality Rate:  {:.{}f}".format( datewise["Mortality Rate"].median(), precision ))     

print( "Average Recovery Rate   {:.{}f}".format( datewise["Recovery Rate"].mean(), precision )) 

#print( "Median Recovery Rate:   {:.{}f}".format( datewise["Recovery Rate"].median(), precision ))



plt.savefig('005pl.png')
plt.figure(figsize=(25,8))

plt.plot(datewise["Confirmed"].diff().fillna(0),label="Daily increase in Confiremd Cases",linewidth=3)

plt.plot(datewise["Recovered"].diff().fillna(0),label="Daily increase in Recovered Cases",linewidth=3)

plt.plot(datewise["Deaths"].diff().fillna(0),label="Daily increase in Death Cases",linewidth=3)

plt.xlabel("dates")

plt.ylabel("Daily Increment")

plt.title("Daily increase in different Types of Cases Worldwide")

plt.xticks(rotation=90)

plt.legend()



print("Daily increase in different Types of Cases Worldwide")

print("Average increase in number of Confirmed Cases every day: ",np.round(datewise["Confirmed"].diff().fillna(0).mean()))

print("Average increase in number of Recovered Cases every day: ",np.round(datewise["Recovered"].diff().fillna(0).mean()))

print("Average increase in number of Deaths Cases every day:     ",np.round(datewise["Deaths"].diff().fillna(0).mean()))



plt.savefig('006pl.png')
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
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(25,8))

top_15_confirmed=countrywise.sort_values(["Confirmed"],ascending=False).head(15)

top_15_deaths=countrywise.sort_values(["Deaths"],ascending=False).head(15)

sns.barplot(x=top_15_confirmed["Confirmed"],y=top_15_confirmed.index,ax=ax1)

ax1.set_title("Top 15 countries as per Number of Confirmed Cases")

sns.barplot(x=top_15_deaths["Deaths"],y=top_15_deaths.index,ax=ax2)

ax2.set_title("Top 15 countries as per Number of Death Cases")



plt.savefig('007pl.png')
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(25,8))

countrywise_plot_mortal=countrywise[countrywise["Confirmed"]>500].sort_values(["Mortality"],ascending=False).head(15)

sns.barplot(x=countrywise_plot_mortal["Mortality"],y=countrywise_plot_mortal.index,ax=ax1)

ax1.set_title("Top 15 Countries according High Mortatlity Rate")

ax1.set_xlabel("Mortality (in Percentage)")

countrywise_plot_recover=countrywise[countrywise["Confirmed"]>500].sort_values(["Recovery"],ascending=False).head(15)

sns.barplot(x=countrywise_plot_recover["Recovery"],y=countrywise_plot_recover.index, ax=ax2)

ax2.set_title("Top 15 Countries according High Recovery Rate")

ax2.set_xlabel("Recovery (in Percentage)")



plt.savefig('008pl.png')
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(25,8))

countrywise_plot_mortal=countrywise[countrywise["Confirmed"]>500].sort_values(["Mortality"],ascending=False).tail(15)

sns.barplot(x=countrywise_plot_mortal["Mortality"],y=countrywise_plot_mortal.index,ax=ax1)

ax1.set_title("Top 15 Countries according Low Mortatlity Rate")

ax1.set_xlabel("Mortality (in Percentage)")

countrywise_plot_recover=countrywise[countrywise["Confirmed"]>500].sort_values(["Recovery"],ascending=False).tail(15)

sns.barplot(x=countrywise_plot_recover["Recovery"],y=countrywise_plot_recover.index, ax=ax2)

ax2.set_title("Top 15 Countries according Low Recovery Rate")

ax2.set_xlabel("Recovery (in Percentage)")



plt.savefig('009pl.png')
no_recovered_countries=countrywise[(countrywise["Confirmed"]>50)&(countrywise["Recovered"]==0)][["Confirmed","Deaths"]]

no_recovered_countries["Mortality Rate"]=(no_recovered_countries["Deaths"]/no_recovered_countries["Confirmed"])*100

no_recovered_countries[no_recovered_countries["Mortality Rate"]>0].sort_values(["Mortality Rate"],ascending=False)
no_deaths=countrywise[(countrywise["Confirmed"]>100)&(countrywise["Deaths"]==0)]

no_deaths[no_deaths["Recovery"]>0].sort_values(["Recovery"],ascending=False).drop(["Mortality"],1)
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(25,8))

countrywise["Active Cases"]=(countrywise["Confirmed"]-countrywise["Recovered"]-countrywise["Deaths"])

countrywise["Outcome Cases"]=(countrywise["Recovered"]+countrywise["Deaths"])

top_15_active=countrywise.sort_values(["Active Cases"],ascending=False).head(15)

top_15_outcome=countrywise.sort_values(["Outcome Cases"],ascending=False).head(15)

sns.barplot(x=top_15_active["Active Cases"],y=top_15_active.index,ax=ax1)

sns.barplot(x=top_15_outcome["Outcome Cases"],y=top_15_outcome.index,ax=ax2)

ax1.set_title("Top 15 Countries with Most Number of Active Cases")

ax2.set_title("Top 15 Countries with Most Number of Closed Cases")



plt.savefig('010pl.png')
country_date=covid.groupby(["Country/Region","ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

confirm_rate=[]

for country in countrywise.index:

    days=country_date.ix[country].shape[0]

    confirm_rate.append((countrywise.ix[country]["Confirmed"])/days)

countrywise["Confirm Cases/Day"]=confirm_rate
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(25,8))

top_15_ccpd=countrywise.sort_values(["Confirm Cases/Day"],ascending=False).head(15)

sns.barplot(y=top_15_ccpd.index,x=top_15_ccpd["Confirm Cases/Day"],ax=ax1)

ax1.set_title("Top 15 countries as per high number Confirmed Cases per Day")

bottom_15_ccpd=countrywise[countrywise["Confirmed"]>1000].sort_values(["Confirm Cases/Day"],ascending=False).tail(15)

sns.barplot(y=bottom_15_ccpd.index,x=bottom_15_ccpd["Confirm Cases/Day"],ax=ax2)

ax2.set_title("Top 15 countries as per Lowest Confirmed Cases per Day having more than 1000 Confirmed Cases")



plt.savefig('011pl.png')
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(25,8))

countrywise["Survival Probability"]=(1-(countrywise["Deaths"]/countrywise["Confirmed"]))*100

top_25_survival=countrywise[countrywise["Confirmed"]>1000].sort_values(["Survival Probability"],ascending=False).head(15)

sns.barplot(x=top_25_survival["Survival Probability"],y=top_25_survival.index,ax=ax1)

ax1.set_title("Top 25 Countries with Maximum Survival Probability having more than 1000 Confiremed Cases")



precision = 2

 



print( "Mean Survival Probability across all countries: {:.{}f}".format( countrywise["Survival Probability"].mean(), precision ))

print( "Median Survival Probability across all countries: {:.{}f}".format( countrywise["Survival Probability"].median(), precision ))

print( "Mean Death Probability across all countries: {:.{}f}".format( 100-countrywise["Survival Probability"].mean(), precision ))

print( "Median Death Probability across all countries: {:.{}f}".format( 100-countrywise["Survival Probability"].median(), precision ))





Bottom_5_countries=countrywise[countrywise["Confirmed"]>100].sort_values(["Survival Probability"],ascending=True).head(15)

sns.barplot(x=Bottom_5_countries["Survival Probability"],y=Bottom_5_countries.index,ax=ax2)

plt.title("Bottom 15 Countries as per Survival Probability")



plt.savefig('012pl.png')
china_data=covid[covid["Country/Region"]=="Mainland China"]

Italy_data=covid[covid["Country/Region"]=="Italy"]

US_data=covid[covid["Country/Region"]=="US"]

poland_data=covid[covid["Country/Region"]=="Poland"]

brazil_data=covid[covid["Country/Region"]=="Brazil"]

rest_of_world=covid[(covid["Country/Region"]!="Mainland China")&(covid["Country/Region"]!="Italy")&(covid["Country/Region"]!="US")&(covid["Country/Region"]!="Spain")]



datewise_china=china_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise_Italy=Italy_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise_US=US_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise_brazil=brazil_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise_poland=poland_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

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
confirmed_covid19 = datewise["Confirmed"]

confirmed_covid19.sum()
datewise_china["Mortality"]=(datewise_china["Deaths"]/datewise_china["Confirmed"])*100

datewise_Italy["Mortality"]=(datewise_Italy["Deaths"]/datewise_Italy["Confirmed"])*100

datewise_US["Mortality"]=(datewise_US["Deaths"]/datewise_US["Confirmed"])*100

datewise_poland["Mortality"]=(datewise_poland["Deaths"]/datewise_poland["Confirmed"])*100

datewise_restofworld["Mortality"]=(datewise_restofworld["Deaths"]/datewise_restofworld["Confirmed"])*100



datewise_china["Recovery"]=(datewise_china["Recovered"]/datewise_china["Confirmed"])*100

datewise_Italy["Recovery"]=(datewise_Italy["Recovered"]/datewise_Italy["Confirmed"])*100

datewise_US["Recovery"]=(datewise_US["Recovered"]/datewise_US["Confirmed"])*100

datewise_poland["Recovery"]=(datewise_poland["Recovered"]/datewise_poland["Confirmed"])*100

datewise_restofworld["Recovery"]=(datewise_restofworld["Recovered"]/datewise_restofworld["Confirmed"])*100
country = "Poland"

poland_data=covid[covid["Country/Region"]=="Poland"]



poland_data=covid[covid["Country/Region"]==country]

datewise_poland=poland_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

print(datewise_poland.iloc[-1])

print("Total Active Cases: ",datewise_poland["Confirmed"].iloc[-1]-datewise_poland["Recovered"].iloc[-1]-datewise_poland["Deaths"].iloc[-1])

print("Total Closed Cases: ",datewise_poland["Recovered"].iloc[-1]+datewise_poland["Deaths"].iloc[-1])
print("COVID19 situation in " +country +" on " + str(today) +".")



print(" ")



print("Total number of Confirmed Cases:                {:.0f} ".format(datewise_poland["Confirmed"].iloc[-1]))

print("Total number of Recovered Cases:                 {:.0f}".format(datewise_poland["Recovered"].iloc[-1]))

print("Total number of Deaths Cases:                     {:.0f}".format(datewise_poland["Deaths"].iloc[-1]))

print("Total number of Active Cases:                   ",int((datewise_poland["Confirmed"].iloc[-1]-datewise_poland["Recovered"].iloc[-1]-datewise_poland["Deaths"].iloc[-1])))

print("Total number of Closed Cases:                   ",int(datewise_poland["Recovered"].iloc[-1]+datewise_poland["Deaths"].iloc[-1]))

print("Number of Confirmed Cases per Day:               ",int(np.round(datewise_poland["Confirmed"].iloc[-1]/datewise_poland.shape[0])))

print("Number of Recovered Cases per Day:                ",int(np.round(datewise_poland["Recovered"].iloc[-1]/datewise_poland.shape[0])))

print("Number of Death Cases per Day:                    ",int(np.round(datewise_poland["Deaths"].iloc[-1]/datewise_poland.shape[0])))

print("Number of Confirmed Cases per hour:                ",int(np.round(datewise_poland["Confirmed"].iloc[-1]/((datewise.shape[0])*24))))

print("Number of Recovered Cases per hour:                ",int(np.round(datewise_poland["Recovered"].iloc[-1]/((datewise_poland.shape[0])*24))))

#print("Number of Death Cases per hour:                ",int(np.round(datewise_poland["Deaths"].iloc[-1]/((datewise_poland.shape[0])*24))))



print(" ")

print("Acknowledgements:")

print("Thanks to the WHO and Johns Hopkins University for making the ")

print("data available for educational and academic research purposes - Jair Ribeiro")
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

#print( "{:.{}f}".format( pi, precision )) 



print("Average Mortality and Recovery Rates in " + country + ".(between parenthesis the rates worldwide rate).")

print()



print( "Average Mortality Rate: {:.{}f}".format( datewise_poland["Mortality Rate"].mean(), precision ) + " ({:.{}f}".format( datewise["Mortality Rate"].mean(), precision )+")") 

#print( "Median Mortality Rate: {:.{}f}".format( datewise_poland["Mortality Rate"].median(), precision ) + " ({:.{}f}".format( datewise["Mortality Rate"].median(), precision )+")")

print( "Average Recovery Rate: {:.{}f}".format( datewise_poland["Recovery Rate"].mean(), precision ) + " ({:.{}f}".format( datewise["Recovery Rate"].mean(), precision )+")")

#print( "Median Recovery Rate: {:.{}f}".format( datewise_poland["Recovery Rate"].median(), precision ) + " ({:.{}f}".format( datewise["Recovery Rate"].median(), precision )+")")
fig, (ax1,ax2) = plt.subplots(2, 1,figsize=(20,10))

ax1.plot(datewise_poland["Confirmed"],marker='o',label="Confirmed Cases")

ax1.plot(datewise_poland["Recovered"],marker='*',label="Recovered Cases")

ax1.plot(datewise_poland["Deaths"],marker='^',label="Death Cases")

ax1.set_ylabel("Number of Patients")

ax1.set_xlabel("Date")

ax1.legend()

ax1.set_title("Growth Rate Plot for different Types of cases in " + country +".")

for tick in ax1.get_xticklabels():

    tick.set_rotation(90)

sns.barplot(datewise_poland.index.date,datewise_poland["Confirmed"]-datewise_poland["Recovered"]-datewise_poland["Deaths"],ax=ax2)

ax2.set_xlabel("Date")

ax2.set_ylabel("Number of Active Cases")

ax2.set_title("Distribution of Number of Active Cases over Date")

for tick in ax2.get_xticklabels():

    tick.set_rotation(90)

    

plt.savefig('015pl.png')
plt.figure(figsize=(25,8))

plt.plot(datewise_poland["Confirmed"].diff().fillna(0),label="Daily increase in Confiremd Cases",linewidth=3)

plt.plot(datewise_poland["Recovered"].diff().fillna(0),label="Daily increase in Recovered Cases",linewidth=3)

plt.plot(datewise_poland["Deaths"].diff().fillna(0),label="Daily increase in Death Cases",linewidth=3)

plt.xlabel("dates")

plt.ylabel("Daily Increment")

plt.title("Daily increase in different Types of Cases Worldwide")

plt.xticks(rotation=90)

plt.legend()



print("Daily increase in different Types of Cases in " + country +".")

print()

print("Average increase in number of Confirmed Cases every day: ",np.round(datewise_poland["Confirmed"].diff().fillna(0).mean()))

print("Average increase in number of Recovered Cases every day:  ",np.round(datewise_poland["Recovered"].diff().fillna(0).mean()))

print("Average increase in number of Deaths Cases every day:     ",np.round(datewise_poland["Deaths"].diff().fillna(0).mean()))



plt.savefig('016pl.png')
poland_increase_confirm=[]

poland_increase_recover=[]

poland_increase_deaths=[]

for i in range(datewise_poland.shape[0]-1):

    poland_increase_confirm.append(((datewise_poland["Confirmed"].iloc[i+1])/datewise_poland["Confirmed"].iloc[i]))

    poland_increase_recover.append(((datewise_poland["Recovered"].iloc[i+1])/datewise_poland["Recovered"].iloc[i]))

    poland_increase_deaths.append(((datewise_poland["Deaths"].iloc[i+1])/datewise_poland["Deaths"].iloc[i]))

poland_increase_confirm.insert(0,1)

poland_increase_recover.insert(0,1)

poland_increase_deaths.insert(0,1)



plt.figure(figsize=(25,8))

plt.plot(datewise_poland.index,poland_increase_confirm,label="Growth Factor of Confirmed Cases",linewidth=3)

plt.plot(datewise_poland.index,poland_increase_recover,label="Growth Factor of Recovered Cases",linewidth=3)

plt.plot(datewise_poland.index,poland_increase_deaths,label="Growth Factor of Death Cases",linewidth=3)

plt.axhline(1,linestyle='--',color="black",label="Baseline")

plt.xticks(rotation=90)

plt.title("Datewise Growth Factor of different Types of Cases in " + country +".")

plt.ylabel("Growth Rate")

plt.xlabel("Date")

plt.legend()



plt.savefig('017pl.png')
plt.figure(figsize=(25,8))

plt.plot(datewise_poland["Confirmed"].diff().fillna(0),linewidth=3)

plt.plot(datewise_poland["Recovered"].diff().fillna(0),linewidth=3)

plt.plot(datewise_poland["Deaths"].diff().fillna(0),linewidth=3)

plt.ylabel("Number of Confirmed Cases")

plt.xlabel("Date")

plt.title("Daily increase in Number of Confirmed Cases in " + country +".")

plt.xticks(rotation=90)



plt.savefig('018pl.png')
plt.figure(figsize=(25,8))

plt.plot(datewise_poland["Confirmed"].diff().fillna(0),label="Daily increase in Confiremd Cases",linewidth=3)

plt.plot(datewise_poland["Recovered"].diff().fillna(0),label="Daily increase in Recovered Cases",linewidth=3)

plt.plot(datewise_poland["Deaths"].diff().fillna(0),label="Daily increase in Death Cases",linewidth=3)

plt.xlabel("dates")

plt.ylabel("Daily Increment")

plt.title("Daily increase in different Types of Cases Worldwide")

plt.xticks(rotation=90)

plt.legend()



print("Average increase in number of Confirmed Cases every day: ",np.round(datewise_poland["Confirmed"].diff().fillna(0).mean()))

print("Average increase in number of Recovered Cases every day:  ",np.round(datewise_poland["Recovered"].diff().fillna(0).mean()))

print("Average increase in number of Deaths Cases every day:     ",np.round(datewise_poland["Deaths"].diff().fillna(0).mean()))



plt.savefig('019pl.png')
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

    

plt.figure(figsize=(25,8))

plt.plot(week_num_poland,poland_weekwise_confirmed,linewidth=3,label="Weekly Growth of Confirmed Cases")

plt.plot(week_num_poland,poland_weekwise_recovered,linewidth=3,label="Weekly Growth of Recovered Cases")

plt.plot(week_num_poland,poland_weekwise_deaths,linewidth=3,label="Weekly Growth of Death Cases")

plt.xlabel('Week Number')

plt.ylabel("Number of Cases")

plt.title("Weekly Growth of different types of Cases in " + country +".")

plt.legend()



plt.savefig('020pl.png')
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
#poland_data=covid[covid["Country/Region"]=="Poland"]

datewise_poland=poland_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

print(datewise_poland.iloc[-1])

print("Total Active Cases: ",datewise_poland["Confirmed"].iloc[-1]-datewise_poland["Recovered"].iloc[-1]-datewise_poland["Deaths"].iloc[-1])

print("Total Closed Cases: ",datewise_poland["Recovered"].iloc[-1]+datewise_poland["Deaths"].iloc[-1])
datewise = datewise_poland

datewise["Days Since"]=datewise.index-datewise.index[0]

datewise["Days Since"]=datewise["Days Since"].dt.days
train_ml=datewise.iloc[:int(datewise.shape[0]*0.90)]

valid_ml=datewise.iloc[int(datewise.shape[0]*0.90):]

model_scores=[]
lin_reg=LinearRegression(normalize=True)
lin_reg.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))
prediction_valid_linreg=lin_reg.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
model_scores.append(np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_linreg)))

print("Root Mean Square Error for Linear Regression: ",np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_linreg)))
plt.figure(figsize=(25,8))

prediction_linreg=lin_reg.predict(np.array(datewise["Days Since"]).reshape(-1,1))

plt.plot(datewise["Confirmed"],label="Actual Confirmed Cases")

plt.plot(datewise.index,prediction_linreg, linestyle='--',label="Predicted Confirmed Cases using Linear Regression",color='black')

plt.xlabel('Time')

plt.ylabel('Confirmed Cases')

plt.title("Confirmed Cases Linear Regression Prediction in " + country +".")

plt.xticks(rotation=90)

plt.legend()
#Intializing SVR Model and with hyperparameters for GridSearchCV

svm=SVR(C=1,degree=6,kernel='poly',epsilon=0.01)
#Performing GridSearchCV to find the Best Estimator

svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))
prediction_valid_svm=svm.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
model_scores.append(np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_svm)))



precision = 2

#print( "{:.{}f}".format( pi, precision )) 



print( "Root Mean Square Error for Support Vectore Machine: {:.{}f}".format( np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_svm)), precision ))
plt.figure(figsize=(25,8))

prediction_svm=svm.predict(np.array(datewise["Days Since"]).reshape(-1,1))

plt.plot(datewise["Confirmed"],label="Train Confirmed Cases",linewidth=3)

plt.plot(datewise.index,prediction_svm, linestyle='--',label="Best Fit for SVR",color='black')

plt.xlabel('Time')

plt.ylabel('Confirmed Cases')

plt.title("Confirmed Cases Support Vector Machine Regressor Prediction in " + country +".")

plt.xticks(rotation=90)

plt.legend()
new_date=[]

new_prediction_lr=[]

new_prediction_svm=[]

for i in range(1,18):

    new_date.append(datewise.index[-1]+timedelta(days=i))

    new_prediction_lr.append(lin_reg.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0][0])

    new_prediction_svm.append(svm.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0])
pd.set_option('precision', 0)

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('display.float_format', lambda x: '%.6f' % x)

model_predictions=pd.DataFrame(zip(new_date,new_prediction_lr,new_prediction_svm),columns=["Dates","Linear Regression Prediction","SVM Prediction"])

#model_predictions.head()
df = pd.DataFrame(zip(new_date,new_prediction_lr,new_prediction_svm),columns=["Dates","Linear Regression Prediction","SVM Prediction"])



df['Dates'] = pd.to_datetime(df['Dates'])

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('precision', 0)



#df = df[(df['yhat']>0)]

df.rename(columns={'Linear Regression Prediction': 'LRP', 'SVM Prediction': 'SVM'}, inplace=True)

#df
model_train=datewise.iloc[:int(datewise.shape[0]*0.90)]

valid=datewise.iloc[int(datewise.shape[0]*0.90):]
holt=Holt(np.asarray(model_train["Confirmed"])).fit(smoothing_level=1.3, smoothing_slope=0.9)

y_pred=valid.copy()
y_pred["Holt"]=holt.forecast(len(valid))

model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt"])))



precision = 2

#print( "{:.{}f}".format( pi, precision )) 



print( "Root Mean Square Error Holt's Linear Model: {:.{}f}".format( np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt"])), precision ))
plt.figure(figsize=(25,8))

plt.plot(model_train.Confirmed,label="Train Set",marker='o')

valid.Confirmed.plot(label="Validation Set",marker='*')

y_pred.Holt.plot(label="Holt's Linear Model Predicted Set",marker='^')

plt.ylabel("Confirmed Cases")

plt.xlabel("Date Time")

plt.title("Confirmed Holt's Linear Model Prediction in " + country +".")

plt.xticks(rotation=90)

plt.legend()
holt_new_date=[]

holt_new_prediction=[]

for i in range(1,18):

    holt_new_date.append(datewise.index[-1]+timedelta(days=i))

    holt_new_prediction.append(holt.forecast((len(valid)+i))[-1])



model_predictions["Holts Linear Model Prediction"]=holt_new_prediction

#model_predictions.head()
df = pd.DataFrame(model_predictions,columns=["Dates","Linear Regression Prediction","SVM Prediction","Holts Linear Model Prediction"])



df['Dates'] = pd.to_datetime(df['Dates'])

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('precision', 0)



#df = df[(df['yhat']>0)]

df.rename(columns={'Linear Regression Prediction': 'LRP', 'SVM Prediction': 'SVM','Holts Linear Model Prediction': 'Holts'}, inplace=True)

#df
model_train=datewise.iloc[:int(datewise.shape[0]*0.90)]

valid=datewise.iloc[int(datewise.shape[0]*0.90):]

y_pred=valid.copy()
es=ExponentialSmoothing(np.asarray(model_train['Confirmed']),seasonal_periods=5,trend='add', seasonal='add').fit()
y_pred["Holt's Winter Model"]=es.forecast(len(valid))
model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt's Winter Model"])))



precision = 2

#print( "{:.{}f}".format( pi, precision )) 



print( "Root Mean Square Error for Holt's Winter Model: {:.{}f}".format( np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt's Winter Model"])), precision ))
plt.figure(figsize=(25,8))

plt.plot(model_train.Confirmed,label="Train Set",marker='o')

valid.Confirmed.plot(label="Validation Set",marker='*')

y_pred["Holt\'s Winter Model"].plot(label="Holt's Winter Model Predicted Set",marker='^')

plt.ylabel("Confirmed Cases")

plt.xlabel("Date Time")

plt.title("Confirmedd Cases Holt's Winter Model Prediction")

plt.xticks(rotation=90)

plt.legend()
holt_winter_new_prediction=[]

for i in range(1,18):

    holt_winter_new_prediction.append(es.forecast((len(valid)+i))[-1])

model_predictions["Holts Winter Model Prediction"]=holt_winter_new_prediction

model_predictions.head()

#model_predictions
df = pd.DataFrame(model_predictions,columns=["Dates","Linear Regression Prediction","SVM Prediction"\

                                             ,"Holts Linear Model Prediction","Holts Winter Model Prediction"])



df['Dates'] = pd.to_datetime(df['Dates'])

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('precision', 0)



#df = df[(df['yhat']>0)]

df.rename(columns={'Linear Regression Prediction': 'LRP', 'SVM Prediction': 'SVM','Holts Linear Model Prediction': 'HLM'\

,'Holts Winter Model Prediction': 'HWM'}, inplace=True)

df
y_pred["Holt\'s Winter Model"].head()
model_train=datewise.iloc[:int(datewise.shape[0]*0.90)]

valid=datewise.iloc[int(datewise.shape[0]*0.90):]

y_pred=valid.copy()
from pandas.plotting import autocorrelation_plot

plt.figure(figsize=(25,8))

autocorrelation_plot(datewise["Confirmed"])
#fig, (ax1,ax2,ax3) = plt.subplots(3, 1,figsize=(11,7))

#import statsmodels.api as sm

#results=sm.tsa.seasonal_decompose(model_train["Confirmed"])

#ax1.plot(results.trend)

#ax2.plot(results.seasonal)

#ax3.plot(results.resid)
print("Results of Dickey-Fuller test for Original Time Series")

dftest = adfuller(model_train["Confirmed"], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

print(dfoutput)
log_series=np.log(model_train["Confirmed"])
prophet_c=Prophet(interval_width=0.95,weekly_seasonality=True,)

prophet_confirmed=pd.DataFrame(zip(list(datewise.index),list(datewise["Confirmed"])),columns=['ds','y'])
prophet_c.fit(prophet_confirmed)
forecast_c=prophet_c.make_future_dataframe(periods=17)

forecast_confirmed=forecast_c.copy()
confirmed_forecast=prophet_c.predict(forecast_c)

#print(confirmed_forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']])
model_scores.append(np.sqrt(mean_squared_error(datewise["Confirmed"],confirmed_forecast['yhat'].head(datewise.shape[0]))))



precision = 2

#print( "{:.{}f}".format( pi, precision )) 



print( "Root Mean Squared Error for Prophet Model: {:.{}f}".format( np.sqrt(mean_squared_error(datewise["Confirmed"],confirmed_forecast['yhat'].head(datewise.shape[0]))), precision ))
print(prophet_c.plot(confirmed_forecast))
print(prophet_c.plot_components(confirmed_forecast))
model_names=["Linear Regression","Support Vector Machine Regressor","Holt's Linear","Holt's Winter Model",

            "Auto Regressive Model (AR)","Moving Average Model (MA)","ARIMA Model","Facebook's Prophet Model"]

pd.DataFrame(zip(model_names,model_scores),columns=["Model Name","Root Mean Squared Error"]).sort_values(["Root Mean Squared Error"])

print(datewise_poland.iloc[-1])
model_predictions["Prophet's Prediction"]=list(confirmed_forecast["yhat"].tail(17))

model_predictions["Prophet's Upper Bound"]=list(confirmed_forecast["yhat_upper"].tail(17))

#model_predictions.head()






df = pd.DataFrame(model_predictions,columns=["Dates","Linear Regression Prediction","SVM Prediction"\

                                             ,"Holts Linear Model Prediction","Holts Winter Model Prediction","Prophet's Prediction","Prophet's Upper Bound"])



df['Dates'] = pd.to_datetime(df['Dates'])

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('precision', 0)



#df = df[(df['yhat']>0)]

df.rename(columns={'Linear Regression Prediction': 'LRP', 'SVM Prediction': 'SVM','Holts Linear Model Prediction': 'HLM'\

,'Holts Winter Model Prediction': 'HWM','Prophet\'s Prediction': 'Prophet','Prophet\'s Upper Bound': 'PUB'}, inplace=True)

forecast_table = df



df
from datetime import date



today = date.today()

#print("Today's date:", today)



start_date = today

end_date = today

df = forecast_table

after_start_date = df["Dates"] >= start_date

before_end_date = df["Dates"] <= end_date

between_two_dates = after_start_date & before_end_date

todays_cases_forecast = df.loc[between_two_dates]



todays_cases_forecast
def closest(lst, K): 

      

    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 

      

# Driver code

Deaths = int(datewise_poland["Confirmed"].iloc[-1])

lst =  [int(todays_cases_forecast.LRP),int(todays_cases_forecast.SVM),int(todays_cases_forecast.HLM)\

        ,int(todays_cases_forecast.HWM),int(todays_cases_forecast.Prophet),int(todays_cases_forecast.PUB)]

K = int(datewise_poland["Confirmed"].iloc[-1])



# Visualization code

print("Real Number of Cases in " + country +":      " + str(Deaths))



print("Closest Prediction on " + str(today) + ":   " + str(int(closest(lst, K))))
df = pd.DataFrame(model_predictions,columns=["Dates","Holts Linear Model Prediction"])



df['Dates'] = pd.to_datetime(df['Dates'])

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('precision', 0)



#df = df[(df['yhat']>0)]

df.rename(columns={'Holts Linear Model Prediction': 'Forecast'}, inplace=True)

forecast_table = df

cases_forecast = df

df
plt.figure(figsize=(25,8))

plt.plot(datewise["Confirmed"],label="Actual Cases")

plt.bar(df.Dates, df.Forecast, color='royalblue', alpha=0.7)





plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.legend(['Confirmed Cases until '+ str(today)])





plt.show()
#poland_data=covid[covid["Country/Region"]=="Poland"]

datewise_poland=poland_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

print(datewise_poland.iloc[-1])

print("Total Active Cases: ",datewise_poland["Confirmed"].iloc[-1]-datewise_poland["Recovered"].iloc[-1]-datewise_poland["Deaths"].iloc[-1])

print("Total Closed Cases: ",datewise_poland["Recovered"].iloc[-1]+datewise_poland["Deaths"].iloc[-1])
datewise = datewise_poland

datewise["Days Since"]=datewise.index-datewise.index[0]

datewise["Days Since"]=datewise["Days Since"].dt.days
train_ml=datewise.iloc[:int(datewise.shape[0]*0.90)]

valid_ml=datewise.iloc[int(datewise.shape[0]*0.90):]

model_scores=[]
lin_reg=LinearRegression(normalize=True)
lin_reg.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Deaths"]).reshape(-1,1))
prediction_valid_linreg=lin_reg.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
model_scores.append(np.sqrt(mean_squared_error(valid_ml["Deaths"],prediction_valid_linreg)))

print("Root Mean Square Error for Linear Regression: ",np.sqrt(mean_squared_error(valid_ml["Deaths"],prediction_valid_linreg)))
plt.figure(figsize=(20,6))

prediction_linreg=lin_reg.predict(np.array(datewise["Days Since"]).reshape(-1,1))

plt.plot(datewise["Deaths"],label="Actual Confirmed Cases")

plt.plot(datewise.index,prediction_linreg, linestyle='--',label="Predicted Deaths using Linear Regression",color='black')

plt.xlabel('Time')

plt.ylabel('Deaths')

plt.title("Deaths Linear Regression Prediction in " + country +".")

plt.xticks(rotation=90)

plt.legend()
#Intializing SVR Model and with hyperparameters for GridSearchCV

svm=SVR(C=1,degree=6,kernel='poly',epsilon=0.01)
#Performing GridSearchCV to find the Best Estimator

svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Deaths"]).reshape(-1,1))
prediction_valid_svm=svm.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
model_scores.append(np.sqrt(mean_squared_error(valid_ml["Deaths"],prediction_valid_svm)))



precision = 2

#print( "{:.{}f}".format( pi, precision )) 



print( "Root Mean Square Error for Support Vectore Machine: {:.{}f}".format( np.sqrt(mean_squared_error(valid_ml["Deaths"],prediction_valid_svm)), precision ))
plt.figure(figsize=(20,6))

prediction_svm=svm.predict(np.array(datewise["Days Since"]).reshape(-1,1))

plt.plot(datewise["Deaths"],label="Train cases of Deaths",linewidth=3)

plt.plot(datewise.index,prediction_svm, linestyle='--',label="Best Fit for SVR",color='black')

plt.xlabel('Time')

plt.ylabel('Deaths')

plt.title("Deaths Support Vector Machine Regressor Prediction in " + country +".")

plt.xticks(rotation=90)

plt.legend()
new_date=[]

new_prediction_lr=[]

new_prediction_svm=[]

for i in range(1,18):

    new_date.append(datewise.index[-1]+timedelta(days=i))

    new_prediction_lr.append(lin_reg.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0][0])

    new_prediction_svm.append(svm.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0])
pd.set_option('precision', 0)

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('display.float_format', lambda x: '%.6f' % x)

model_predictions=pd.DataFrame(zip(new_date,new_prediction_lr,new_prediction_svm),columns=["Dates","Linear Regression Prediction","SVM Prediction"])

#model_predictions.head()
df = pd.DataFrame(zip(new_date,new_prediction_lr,new_prediction_svm),columns=["Dates","Linear Regression Prediction","SVM Prediction"])



df['Dates'] = pd.to_datetime(df['Dates'])

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('precision', 0)



#df = df[(df['yhat']>0)]

df.rename(columns={'Linear Regression Prediction': 'LRP', 'SVM Prediction': 'SVM'}, inplace=True)

#df
model_train=datewise.iloc[:int(datewise.shape[0]*0.90)]

valid=datewise.iloc[int(datewise.shape[0]*0.90):]
holt=Holt(np.asarray(model_train["Deaths"])).fit(smoothing_level=1.3, smoothing_slope=0.9)

y_pred=valid.copy()
y_pred["Holt"]=holt.forecast(len(valid))

model_scores.append(np.sqrt(mean_squared_error(y_pred["Deaths"],y_pred["Holt"])))



precision = 2

#print( "{:.{}f}".format( pi, precision )) 



print( "Root Mean Square Error Holt's Linear Model: {:.{}f}".format( np.sqrt(mean_squared_error(y_pred["Deaths"],y_pred["Holt"])), precision ))
plt.figure(figsize=(20,5))

plt.plot(model_train.Deaths,label="Train Set",marker='o')

valid.Deaths.plot(label="Validation Set",marker='*')

y_pred.Holt.plot(label="Holt's Linear Model Predicted Set",marker='^')

plt.ylabel("Deaths")

plt.xlabel("Date Time")

plt.title("Deaths Holt's Linear Model Prediction in " + country +".")

plt.xticks(rotation=90)

plt.legend()
holt_new_date=[]

holt_new_prediction=[]

for i in range(1,18):

    holt_new_date.append(datewise.index[-1]+timedelta(days=i))

    holt_new_prediction.append(holt.forecast((len(valid)+i))[-1])



model_predictions["Holts Linear Model Prediction"]=holt_new_prediction

#model_predictions.head()
df = pd.DataFrame(model_predictions,columns=["Dates","Linear Regression Prediction","SVM Prediction","Holts Linear Model Prediction"])



df['Dates'] = pd.to_datetime(df['Dates'])

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('precision', 0)



#df = df[(df['yhat']>0)]

df.rename(columns={'Linear Regression Prediction': 'LRP', 'SVM Prediction': 'SVM','Holts Linear Model Prediction': 'Holts'}, inplace=True)

#df
model_train=datewise.iloc[:int(datewise.shape[0]*0.90)]

valid=datewise.iloc[int(datewise.shape[0]*0.90):]

y_pred=valid.copy()
es=ExponentialSmoothing(np.asarray(model_train['Deaths']),seasonal_periods=5,trend='add', seasonal='add').fit()
y_pred["Holt's Winter Model"]=es.forecast(len(valid))
model_scores.append(np.sqrt(mean_squared_error(y_pred["Deaths"],y_pred["Holt's Winter Model"])))



precision = 2

#print( "{:.{}f}".format( pi, precision )) 



print( "Root Mean Square Error for Holt's Winter Model: {:.{}f}".format( np.sqrt(mean_squared_error(y_pred["Deaths"],y_pred["Holt's Winter Model"])), precision ))
plt.figure(figsize=(20,5))

plt.plot(model_train.Deaths,label="Train Set",marker='o')

valid.Deaths.plot(label="Validation Set",marker='*')

y_pred["Holt\'s Winter Model"].plot(label="Holt's Winter Model Predicted Set",marker='^')

plt.ylabel("Deaths")

plt.xlabel("Date Time")

plt.title("Deaths Cases Holt's Winter Model Prediction")

plt.xticks(rotation=90)

plt.legend()
holt_winter_new_prediction=[]

for i in range(1,18):

    holt_winter_new_prediction.append(es.forecast((len(valid)+i))[-1])

model_predictions["Holts Winter Model Prediction"]=holt_winter_new_prediction

model_predictions.head()

#model_predictions
df = pd.DataFrame(model_predictions,columns=["Dates","Linear Regression Prediction","SVM Prediction"\

                                             ,"Holts Linear Model Prediction","Holts Winter Model Prediction"])



df['Dates'] = pd.to_datetime(df['Dates'])

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('precision', 0)



#df = df[(df['yhat']>0)]

df.rename(columns={'Linear Regression Prediction': 'LRP', 'SVM Prediction': 'SVM','Holts Linear Model Prediction': 'HLM'\

,'Holts Winter Model Prediction': 'HWM'}, inplace=True)

df
y_pred["Holt\'s Winter Model"].head()
model_train=datewise.iloc[:int(datewise.shape[0]*0.90)]

valid=datewise.iloc[int(datewise.shape[0]*0.90):]

y_pred=valid.copy()
from pandas.plotting import autocorrelation_plot

plt.figure(figsize=(20, 5))

autocorrelation_plot(datewise["Deaths"])
#fig, (ax1,ax2,ax3) = plt.subplots(3, 1,figsize=(11,7))

#import statsmodels.api as sm

#results=sm.tsa.seasonal_decompose(model_train["Confirmed"])

#ax1.plot(results.trend)

#ax2.plot(results.seasonal)

#ax3.plot(results.resid)
print("Results of Dickey-Fuller test for Original Time Series")

dftest = adfuller(model_train["Deaths"], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

print(dfoutput)
log_series=np.log(model_train["Deaths"])
prophet_c=Prophet(interval_width=0.95,weekly_seasonality=True,)

prophet_Deaths=pd.DataFrame(zip(list(datewise.index),list(datewise["Deaths"])),columns=['ds','y'])
prophet_c.fit(prophet_Deaths)
forecast_c=prophet_c.make_future_dataframe(periods=17)

forecast_Deaths=forecast_c.copy()
Deaths_forecast=prophet_c.predict(forecast_c)

#print(confirmed_forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']])
model_scores.append(np.sqrt(mean_squared_error(datewise["Deaths"],Deaths_forecast['yhat'].head(datewise.shape[0]))))



precision = 2

#print( "{:.{}f}".format( pi, precision )) 



print( "Root Mean Squared Error for Prophet Model: {:.{}f}".format( np.sqrt(mean_squared_error(datewise["Deaths"],Deaths_forecast['yhat'].head(datewise.shape[0]))), precision ))
print(prophet_c.plot(Deaths_forecast))
print(prophet_c.plot_components(Deaths_forecast))
model_names=["Linear Regression","Support Vector Machine Regressor","Holt's Linear","Holt's Winter Model",

            "Auto Regressive Model (AR)","Moving Average Model (MA)","ARIMA Model","Facebook's Prophet Model"]

pd.DataFrame(zip(model_names,model_scores),columns=["Model Name","Root Mean Squared Error"]).sort_values(["Root Mean Squared Error"])

print(datewise_poland.iloc[-1])
model_predictions["Prophet's Prediction"]=list(Deaths_forecast["yhat"].tail(17))

model_predictions["Prophet's Upper Bound"]=list(Deaths_forecast["yhat_upper"].tail(17))

#model_predictions.head()
df = pd.DataFrame(model_predictions,columns=["Dates","Linear Regression Prediction","SVM Prediction"\

                                             ,"Holts Linear Model Prediction","Holts Winter Model Prediction","Prophet's Prediction","Prophet's Upper Bound"])



df['Dates'] = pd.to_datetime(df['Dates'])

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('precision', 0)



#df = df[(df['yhat']>0)]

df.rename(columns={'Linear Regression Prediction': 'LRP', 'SVM Prediction': 'SVM','Holts Linear Model Prediction': 'HLM'\

,'Holts Winter Model Prediction': 'HWM','Prophet\'s Prediction': 'Prophet','Prophet\'s Upper Bound': 'PUB'}, inplace=True)

forecast_table = df



df
from datetime import date



today = date.today()

#print("Today's date:", today)



start_date = today

end_date = today

df = forecast_table

after_start_date = df["Dates"] >= start_date

before_end_date = df["Dates"] <= end_date

between_two_dates = after_start_date & before_end_date

todays_deaths_forecast = df.loc[between_two_dates]



todays_deaths_forecast
def closest(lst, K): 

      

    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 

      

# Driver code

Deaths = int(datewise_poland["Deaths"].iloc[-1])

lst =  [int(todays_deaths_forecast.LRP),int(todays_deaths_forecast.SVM),int(todays_deaths_forecast.HLM)\

        ,int(todays_deaths_forecast.HWM),int(todays_deaths_forecast.Prophet),int(todays_deaths_forecast.PUB)]

K = int(datewise_poland["Deaths"].iloc[-1])



# Visualization code

print("Real Number of Deaths in " + country +":    " + str(Deaths))



print("Closest Prediction on " + str(today) + ":   " + str(int(closest(lst, K))))
df = pd.DataFrame(model_predictions,columns=["Dates","Prophet's Prediction"])



df['Dates'] = pd.to_datetime(df['Dates'])

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('precision', 0)



#df = df[(df['yhat']>0)]

df.rename(columns={'Prophet\'s Prediction': 'Deaths Forecast'}, inplace=True)

forecast_table = df

deaths_forecast = df

df
#poland_data=covid[covid["Country/Region"]=="Poland"]

datewise_poland=poland_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

print(datewise_poland.iloc[-1])

print("Total Active Cases: ",datewise_poland["Confirmed"].iloc[-1]-datewise_poland["Recovered"].iloc[-1]-datewise_poland["Deaths"].iloc[-1])

print("Total Closed Cases: ",datewise_poland["Recovered"].iloc[-1]+datewise_poland["Deaths"].iloc[-1])
datewise = datewise_poland

datewise["Days Since"]=datewise.index-datewise.index[0]

datewise["Days Since"]=datewise["Days Since"].dt.days
train_ml=datewise.iloc[:int(datewise.shape[0]*0.90)]

valid_ml=datewise.iloc[int(datewise.shape[0]*0.90):]

model_scores=[]
lin_reg=LinearRegression(normalize=True)
lin_reg.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Recovered"]).reshape(-1,1))
prediction_valid_linreg=lin_reg.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
model_scores.append(np.sqrt(mean_squared_error(valid_ml["Recovered"],prediction_valid_linreg)))

print("Root Mean Square Error for Linear Regression: ",np.sqrt(mean_squared_error(valid_ml["Recovered"],prediction_valid_linreg)))
plt.figure(figsize=(20,6))

prediction_linreg=lin_reg.predict(np.array(datewise["Days Since"]).reshape(-1,1))

plt.plot(datewise["Recovered"],label="Actual Recovered Cases")

plt.plot(datewise.index,prediction_linreg, linestyle='--',label="Predicted Recoverings using Linear Regression",color='black')

plt.xlabel('Time')

plt.ylabel('Recovered')

plt.title("Recovered Linear Regression Prediction in " + country +".")

plt.xticks(rotation=90)

plt.legend()
#Intializing SVR Model and with hyperparameters for GridSearchCV

svm=SVR(C=1,degree=6,kernel='poly',epsilon=0.01)
#Performing GridSearchCV to find the Best Estimator

svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Recovered"]).reshape(-1,1))
prediction_valid_svm=svm.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
model_scores.append(np.sqrt(mean_squared_error(valid_ml["Recovered"],prediction_valid_svm)))



precision = 2

#print( "{:.{}f}".format( pi, precision )) 



print( "Root Mean Square Error for Support Vectore Machine: {:.{}f}".format( np.sqrt(mean_squared_error(valid_ml["Recovered"],prediction_valid_svm)), precision ))
plt.figure(figsize=(20,6))

prediction_svm=svm.predict(np.array(datewise["Days Since"]).reshape(-1,1))

plt.plot(datewise["Recovered"],label="Train cases of Recovered cases",linewidth=3)

plt.plot(datewise.index,prediction_svm, linestyle='--',label="Best Fit for SVR",color='black')

plt.xlabel('Time')

plt.ylabel('Recovered')

plt.title("Recovered Support Vector Machine Regressor Prediction in " + country +".")

plt.xticks(rotation=90)

plt.legend()
new_date=[]

new_prediction_lr=[]

new_prediction_svm=[]

for i in range(1,18):

    new_date.append(datewise.index[-1]+timedelta(days=i))

    new_prediction_lr.append(lin_reg.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0][0])

    new_prediction_svm.append(svm.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0])
pd.set_option('precision', 0)

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('display.float_format', lambda x: '%.6f' % x)

model_predictions=pd.DataFrame(zip(new_date,new_prediction_lr,new_prediction_svm),columns=["Dates","Linear Regression Prediction","SVM Prediction"])

#model_predictions.head()
df = pd.DataFrame(zip(new_date,new_prediction_lr,new_prediction_svm),columns=["Dates","Linear Regression Prediction","SVM Prediction"])



df['Dates'] = pd.to_datetime(df['Dates'])

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('precision', 0)



#df = df[(df['yhat']>0)]

df.rename(columns={'Linear Regression Prediction': 'LRP', 'SVM Prediction': 'SVM'}, inplace=True)

#df
model_train=datewise.iloc[:int(datewise.shape[0]*0.90)]

valid=datewise.iloc[int(datewise.shape[0]*0.90):]
holt=Holt(np.asarray(model_train["Recovered"])).fit(smoothing_level=1.3, smoothing_slope=0.9)

y_pred=valid.copy()
y_pred["Holt"]=holt.forecast(len(valid))

model_scores.append(np.sqrt(mean_squared_error(y_pred["Recovered"],y_pred["Holt"])))



precision = 2

#print( "{:.{}f}".format( pi, precision )) 



print( "Root Mean Square Error Holt's Linear Model: {:.{}f}".format( np.sqrt(mean_squared_error(y_pred["Recovered"],y_pred["Holt"])), precision ))
plt.figure(figsize=(20,5))

plt.plot(model_train.Recovered,label="Train Set",marker='o')

valid.Recovered.plot(label="Validation Set",marker='*')

y_pred.Holt.plot(label="Holt's Linear Model Predicted Set",marker='^')

plt.ylabel("Recovered")

plt.xlabel("Date Time")

plt.title("Recovered Holt's Linear Model Prediction in " + country +".")

plt.xticks(rotation=90)

plt.legend()
holt_new_date=[]

holt_new_prediction=[]

for i in range(1,18):

    holt_new_date.append(datewise.index[-1]+timedelta(days=i))

    holt_new_prediction.append(holt.forecast((len(valid)+i))[-1])



model_predictions["Holts Linear Model Prediction"]=holt_new_prediction

#model_predictions.head()
df = pd.DataFrame(model_predictions,columns=["Dates","Linear Regression Prediction","SVM Prediction","Holts Linear Model Prediction"])



df['Dates'] = pd.to_datetime(df['Dates'])

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('precision', 0)



#df = df[(df['yhat']>0)]

df.rename(columns={'Linear Regression Prediction': 'LRP', 'SVM Prediction': 'SVM','Holts Linear Model Prediction': 'Holts'}, inplace=True)

#df
model_train=datewise.iloc[:int(datewise.shape[0]*0.90)]

valid=datewise.iloc[int(datewise.shape[0]*0.90):]

y_pred=valid.copy()
es=ExponentialSmoothing(np.asarray(model_train['Recovered']),seasonal_periods=5,trend='add', seasonal='add').fit()
y_pred["Holt's Winter Model"]=es.forecast(len(valid))
model_scores.append(np.sqrt(mean_squared_error(y_pred["Deaths"],y_pred["Holt's Winter Model"])))



precision = 2

#print( "{:.{}f}".format( pi, precision )) 



print( "Root Mean Square Error for Holt's Winter Model: {:.{}f}".format( np.sqrt(mean_squared_error(y_pred["Recovered"],y_pred["Holt's Winter Model"])), precision ))
plt.figure(figsize=(20,5))

plt.plot(model_train.Recovered,label="Train Set",marker='o')

valid.Recovered.plot(label="Validation Set",marker='*')

y_pred["Holt\'s Winter Model"].plot(label="Holt's Winter Model Predicted Set",marker='^')

plt.ylabel("Recovered")

plt.xlabel("Date Time")

plt.title("Recovered Cases Holt's Winter Model Prediction")

plt.xticks(rotation=90)

plt.legend()
holt_winter_new_prediction=[]

for i in range(1,18):

    holt_winter_new_prediction.append(es.forecast((len(valid)+i))[-1])

model_predictions["Holts Winter Model Prediction"]=holt_winter_new_prediction

model_predictions.head()

#model_predictions
df = pd.DataFrame(model_predictions,columns=["Dates","Linear Regression Prediction","SVM Prediction"\

                                             ,"Holts Linear Model Prediction","Holts Winter Model Prediction"])



df['Dates'] = pd.to_datetime(df['Dates'])

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('precision', 0)



#df = df[(df['yhat']>0)]

df.rename(columns={'Linear Regression Prediction': 'LRP', 'SVM Prediction': 'SVM','Holts Linear Model Prediction': 'HLM'\

,'Holts Winter Model Prediction': 'HWM'}, inplace=True)

df
y_pred["Holt\'s Winter Model"].head()
model_train=datewise.iloc[:int(datewise.shape[0]*0.90)]

valid=datewise.iloc[int(datewise.shape[0]*0.90):]

y_pred=valid.copy()
from pandas.plotting import autocorrelation_plot

plt.figure(figsize=(20, 5))

autocorrelation_plot(datewise["Recovered"])
#fig, (ax1,ax2,ax3) = plt.subplots(3, 1,figsize=(11,7))

#import statsmodels.api as sm

#results=sm.tsa.seasonal_decompose(model_train["Confirmed"])

#ax1.plot(results.trend)

#ax2.plot(results.seasonal)

#ax3.plot(results.resid)
print("Results of Dickey-Fuller test for Original Time Series")

dftest = adfuller(model_train["Recovered"], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

print(dfoutput)
log_series=np.log(model_train["Recovered"])
prophet_c=Prophet(interval_width=0.95,weekly_seasonality=True,)

prophet_Recovered=pd.DataFrame(zip(list(datewise.index),list(datewise["Recovered"])),columns=['ds','y'])
prophet_c.fit(prophet_Recovered)
forecast_c=prophet_c.make_future_dataframe(periods=17)

forecast_Recovered=forecast_c.copy()
Recovered_forecast=prophet_c.predict(forecast_c)

#print(confirmed_forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']])
model_scores.append(np.sqrt(mean_squared_error(datewise["Recovered"],Deaths_forecast['yhat'].head(datewise.shape[0]))))



precision = 2

#print( "{:.{}f}".format( pi, precision )) 



print( "Root Mean Squared Error for Prophet Model: {:.{}f}".format( np.sqrt(mean_squared_error(datewise["Recovered"],Recovered_forecast['yhat'].head(datewise.shape[0]))), precision ))
print(prophet_c.plot(Recovered_forecast))
print(prophet_c.plot_components(Recovered_forecast))
model_names=["Linear Regression","Support Vector Machine Regressor","Holt's Linear","Holt's Winter Model",

            "Auto Regressive Model (AR)","Moving Average Model (MA)","ARIMA Model","Facebook's Prophet Model"]

pd.DataFrame(zip(model_names,model_scores),columns=["Model Name","Root Mean Squared Error"]).sort_values(["Root Mean Squared Error"])

print(datewise_poland.iloc[-1])
model_predictions["Prophet's Prediction"]=list(Recovered_forecast["yhat"].tail(17))

model_predictions["Prophet's Upper Bound"]=list(Recovered_forecast["yhat_upper"].tail(17))

#model_predictions.head()
df = pd.DataFrame(model_predictions,columns=["Dates","Linear Regression Prediction","SVM Prediction"\

                                             ,"Holts Linear Model Prediction","Holts Winter Model Prediction","Prophet's Prediction","Prophet's Upper Bound"])



df['Dates'] = pd.to_datetime(df['Dates'])

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('precision', 0)



#df = df[(df['yhat']>0)]

df.rename(columns={'Linear Regression Prediction': 'LRP', 'SVM Prediction': 'SVM','Holts Linear Model Prediction': 'HLM'\

,'Holts Winter Model Prediction': 'HWM','Prophet\'s Prediction': 'Prophet','Prophet\'s Upper Bound': 'PUB'}, inplace=True)

forecast_table = df



df
from datetime import date



today = date.today()

#print("Today's date:", today)



start_date = today

end_date = today

df = forecast_table

after_start_date = df["Dates"] >= start_date

before_end_date = df["Dates"] <= end_date

between_two_dates = after_start_date & before_end_date

todays_recovered_forecast = df.loc[between_two_dates]



todays_recovered_forecast
def closest(lst, K): 

      

    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 

      

# Driver code

Recovered = int(datewise_poland["Recovered"].iloc[-1])

lst =  [int(todays_recovered_forecast.LRP),int(todays_recovered_forecast.SVM),int(todays_recovered_forecast.HLM),int(todays_recovered_forecast.HWM)\

        ,int(todays_recovered_forecast.Prophet),int(todays_recovered_forecast.PUB)]

K = int(datewise_poland["Recovered"].iloc[-1])



# Visualization code

print("Real Number of Recovered Cases in " + country +":    " + str(Recovered))



print("Closest Prediction on " + str(today) + ":            " + str(int(closest(lst, K))))
df = pd.DataFrame(model_predictions,columns=["Dates","Prophet\'s Prediction"])



df['Dates'] = pd.to_datetime(df['Dates'])

pd.options.display.float_format = '{:,.0f}'.format

pd.set_option('precision', 0)



#df = df[(df['yhat']>0)]

df.rename(columns={'Prophet\'s Prediction': 'Recovered'}, inplace=True)

forecast_table = df

recovered_forecast = df



df
complete_forecast = pd.concat([cases_forecast,recovered_forecast, deaths_forecast])



# Stack the DataFrames on top of each other

vertical_stack = pd.concat([cases_forecast,recovered_forecast, deaths_forecast], axis=0)



# Place the DataFrames side by side

horizontal_stack = pd.concat([cases_forecast,recovered_forecast, deaths_forecast], axis=1)
todays_cases_forecast
todays_deaths_forecast
todays_recovered_forecast
forecast_summary = pd.concat([cases_forecast.set_index('Dates'), recovered_forecast.set_index('Dates'), deaths_forecast.set_index('Dates')], axis=1, join='inner')

forecast_summary.rename(columns={'Dates': 'Date','Forecast': 'Cases','Deaths Forecast': 'Deaths'}, inplace=True)
plt.figure(figsize=(25,8))

plt.plot(datewise_poland["Confirmed"],label="Actual Cases")

plt.bar(df.Dates, forecast_summary.Cases, color='royalblue', alpha=0.7)





plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.legend(['Confirmed Cases until '+ str(today)])



plt.savefig('022pl.png')

plt.show()
plt.figure(figsize=(25,8))

plt.plot(datewise_poland["Recovered"],label="Recovered")

plt.bar(df.Dates, forecast_summary.Recovered, color='green', alpha=0.7)





plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.legend(['Recovered Cases until '+ str(today)])



plt.savefig('023pl.png')

plt.show()
plt.figure(figsize=(25,8))

plt.plot(datewise_poland["Deaths"],label="Deaths")

plt.bar(df.Dates, forecast_summary.Deaths, color='red', alpha=0.7)





plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.legend(['Deaths until '+ str(today)])



plt.savefig('024pl.png')

plt.show()


forecast_summary
from IPython.core.display import HTML
# Import

# ======



# essential libraries

import random

from datetime import timedelta



# storing and anaysis

import numpy as np

import pandas as pd



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

import calmap

import folium



# color pallette

cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 



# converter

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()   



# hide warnings

import warnings

warnings.filterwarnings('ignore')



import os



if not os.path.exists("images"):

    os.mkdir("images")
# for offline ploting

# ===================

from plotly.offline import plot, iplot, init_notebook_mode

init_notebook_mode(connected=True)
# list files

# ==========



# !ls ../input/corona-virus-report
# importing datasets

# ==================



full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])

full_table.sample(6)
# dataframe info

# full_table.info()
# checking for missing value

# full_table.isna().sum()
# Ship

# ====



# ship rows

ship_rows = full_table['Province/State'].str.contains('Grand Princess') | full_table['Province/State'].str.contains('Diamond Princess') | full_table['Country/Region'].str.contains('Diamond Princess') | full_table['Country/Region'].str.contains('MS Zaandam')



# ship

ship = full_table[ship_rows]



# full table 

full_table = full_table[~(ship_rows)]



# Latest cases from the ships

ship_latest = ship[ship['Date']==max(ship['Date'])]



# ship_latest.style.background_gradient(cmap='Pastel1_r')
# Cleaning data

# =============



# Active Case = confirmed - deaths - recovered

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']



# replacing Mainland china with just China

full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')



# filling missing values 

full_table[['Province/State']] = full_table[['Province/State']].fillna('')

full_table[['Confirmed', 'Deaths', 'Recovered', 'Active']] = full_table[['Confirmed', 'Deaths', 'Recovered', 'Active']].fillna(0)



# fixing datatypes

full_table['Recovered'] = full_table['Recovered'].astype(int)



full_table.sample(6)
# Grouped by day, country

# =======================



full_grouped = full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()



# new cases ======================================================

temp = full_grouped.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']

temp = temp.sum().diff().reset_index()



mask = temp['Country/Region'] != temp['Country/Region'].shift(1)



temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan

temp.loc[mask, 'Recovered'] = np.nan



# renaming columns

temp.columns = ['Country/Region', 'Date', 'New cases', 'New deaths', 'New recovered']

# =================================================================



# merging new values

full_grouped = pd.merge(full_grouped, temp, on=['Country/Region', 'Date'])



# filling na with 0

full_grouped = full_grouped.fillna(0)



# fixing data types

cols = ['New cases', 'New deaths', 'New recovered']

full_grouped[cols] = full_grouped[cols].astype('int')



full_grouped['New cases'] = full_grouped['New cases'].apply(lambda x: 0 if x<0 else x)



full_grouped.sample(6)
# Day wise

# ========



# table

day_wise = full_grouped.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases'].sum().reset_index()



# number cases per 100 cases

day_wise['Deaths / 100 Cases'] = round((day_wise['Deaths']/day_wise['Confirmed'])*100, 2)

day_wise['Recovered / 100 Cases'] = round((day_wise['Recovered']/day_wise['Confirmed'])*100, 2)

day_wise['Deaths / 100 Recovered'] = round((day_wise['Deaths']/day_wise['Recovered'])*100, 2)



# no. of countries

day_wise['No. of countries'] = full_grouped[full_grouped['Confirmed']!=0].groupby('Date')['Country/Region'].unique().apply(len).values



# fillna by 0

cols = ['Deaths / 100 Cases', 'Recovered / 100 Cases', 'Deaths / 100 Recovered']

day_wise[cols] = day_wise[cols].fillna(0)



day_wise.head()
# Country wise

# ============



# getting latest values

country_wise = full_grouped[full_grouped['Date']==max(full_grouped['Date'])].reset_index(drop=True).drop('Date', axis=1)



# group by country

country_wise = country_wise.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases'].sum().reset_index()



# per 100 cases

country_wise['Deaths / 100 Cases'] = round((country_wise['Deaths']/country_wise['Confirmed'])*100, 2)

country_wise['Recovered / 100 Cases'] = round((country_wise['Recovered']/country_wise['Confirmed'])*100, 2)

country_wise['Deaths / 100 Recovered'] = round((country_wise['Deaths']/country_wise['Recovered'])*100, 2)



cols = ['Deaths / 100 Cases', 'Recovered / 100 Cases', 'Deaths / 100 Recovered']

country_wise[cols] = country_wise[cols].fillna(0)

country_wise.sort_values(by=['New cases'], inplace=True, ascending=False)



country_wise.head()
# load population dataset

pop = pd.read_csv("../input/population-by-country-2020/population_by_country_2020.csv")



# select only population

pop = pop.iloc[:, :2]



# rename column names

pop.columns = ['Country/Region', 'Population']



# merged data

country_wise = pd.merge(country_wise, pop, on='Country/Region', how='left')



# update population

cols = ['Burma', 'Congo (Brazzaville)', 'Congo (Kinshasa)', "Cote d'Ivoire", 'Czechia', 

        'Kosovo', 'Saint Kitts and Nevis', 'Saint Vincent and the Grenadines', 

        'Taiwan*', 'US', 'West Bank and Gaza', 'Poland']

pops = [54409800, 89561403, 5518087, 26378274, 10708981, 1793000, 

        53109, 110854, 23806638, 330541757, 4543126,37854825]

for c, p in zip(cols, pops):

    country_wise.loc[country_wise['Country/Region']== c, 'Population'] = p

    

# missing values

# country_wise.isna().sum()

# country_wise[country_wise['Population'].isna()]['Country/Region'].tolist()



# Cases per population

country_wise['Cases / Million People'] = round((country_wise['Confirmed'] / country_wise['Population']) * 1000000)



country_wise.sort_values(by=['Cases / Million People'], inplace=True, ascending=False)



country_wise.head()
country = "Poland"

#poland_data=covid[covid["Country/Region"]=="Poland"]



poland_data=covid[covid["Country/Region"]==country]

#datewise_poland=poland_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

#print(datewise_poland.iloc[-1])

#print("Total Active Cases: ",datewise_poland["Confirmed"].iloc[-1]-datewise_poland["Recovered"].iloc[-1]-datewise_poland["Deaths"].iloc[-1])

#print("Total Closed Cases: ",datewise_poland["Recovered"].iloc[-1]+datewise_poland["Deaths"].iloc[-1])









# load population dataset

pop = pd.read_csv("../input/population-by-country-2020/population_by_country_2020.csv")



# select only population

pop = pop.iloc[:, :2]



# rename column names

pop.columns = ['Country/Region', 'Population']



# merged data

country_wise = pd.merge(country_wise, pop, on='Country/Region', how='left')



# update population

cols = ['Burma', 'Congo (Brazzaville)', 'Congo (Kinshasa)', "Cote d'Ivoire", 'Czechia', 

        'Kosovo', 'Saint Kitts and Nevis', 'Saint Vincent and the Grenadines', 

        'Taiwan*', 'US', 'West Bank and Gaza', 'Poland']

pops = [54409800, 89561403, 5518087, 26378274, 10708981, 1793000, 

        53109, 110854, 23806638, 330541757, 4543126,37854825]

for c, p in zip(cols, pops):

    country_wise.loc[country_wise['Country/Region']== c, 'Population'] = p

    

# missing values

# country_wise.isna().sum()

# country_wise[country_wise['Population'].isna()]['Country/Region'].tolist()



# Cases per population

country_wise['Cases / Million People'] = round((country_wise['Confirmed'] / country_wise['Population']) * 1000000)



country_wise_local=country_wise[country_wise["Country/Region"]==country]



country_wise_local.head()
print("General Information about the COVID-19 in " +country +" on " + str(today) +".")



print(" ")



print("Number of Confirmed Cases:                {:.0f} ".format(datewise_poland["Confirmed"].iloc[-1]))

print("Number of Recovered Cases:                 {:.0f}".format(datewise_poland["Recovered"].iloc[-1]))

print("Number of Deaths Cases:                     {:.0f}".format(datewise_poland["Deaths"].iloc[-1]))

print("Number of Active Cases:                   ",int((datewise_poland["Confirmed"].iloc[-1]-datewise_poland["Recovered"].iloc[-1]-datewise_poland["Deaths"].iloc[-1])))

print("Number of Closed Cases:                   ",int(datewise_poland["Recovered"].iloc[-1]+datewise_poland["Deaths"].iloc[-1]))

print("Number of Cases / Million People :         ",int(country_wise_local["Cases / Million People"].iloc[-1]))



print(" ")



print("Number of Deaths every 100 cases :           ",int(country_wise_local["Deaths / 100 Cases"].iloc[-1]))

print("Number of Recoveries every 100 cases :      ",int(country_wise_local["Recovered / 100 Cases"].iloc[-1]))

print("Number of Deaths every 100 recoveries :     ",int(country_wise_local["Deaths / 100 Recovered"].iloc[-1]))



print(" ")



print("Confirmed Cases per Day:                   ",int(np.round(datewise_poland["Confirmed"].iloc[-1]/datewise_poland.shape[0])))

print("Recovered Cases per Day:                    ",int(np.round(datewise_poland["Recovered"].iloc[-1]/datewise_poland.shape[0])))

print("Death Cases per Day:                        ",int(np.round(datewise_poland["Deaths"].iloc[-1]/datewise_poland.shape[0])))

print("Confirmed Cases per hour:                    ",int(np.round(datewise_poland["Confirmed"].iloc[-1]/((datewise.shape[0])*24))))

print("Recovered Cases per hour:                    ",int(np.round(datewise_poland["Recovered"].iloc[-1]/((datewise_poland.shape[0])*24))))

#print("Number of Death Cases per hour:                ",int(np.round(datewise_poland["Deaths"].iloc[-1]/((datewise_poland.shape[0])*24))))



print(" ")

print("Acknowledgements:")

print("Thanks to the WHO and Johns Hopkins University for making the ")

print("data available for educational and academic research purposes - Jair Ribeiro")
today = full_grouped[full_grouped['Date']==max(full_grouped['Date'])].reset_index(drop=True).drop('Date', axis=1)[['Country/Region', 'Confirmed']]

last_week = full_grouped[full_grouped['Date']==max(full_grouped['Date'])-timedelta(days=7)].reset_index(drop=True).drop('Date', axis=1)[['Country/Region', 'Confirmed']]



temp = pd.merge(today, last_week, on='Country/Region', suffixes=(' today', ' last week'))



# temp = temp[['Country/Region', 'Confirmed last week']]

temp['1 week change'] = temp['Confirmed today'] - temp['Confirmed last week']



temp = temp[['Country/Region', 'Confirmed last week', '1 week change']]



country_wise = pd.merge(country_wise, temp, on='Country/Region')



country_wise['1 week % increase'] = round(country_wise['1 week change']/country_wise['Confirmed last week']*100, 2)





country_wise.sort_values(by=['Cases / Million People'], inplace=True, ascending=False)



country_wise.head()
temp = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)



tm = temp.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Recovered'])

fig = px.treemap(tm, path=["variable"], values="value", height=225, width=640,

                 color_discrete_sequence=[act, rec, dth])

fig.data[0].textinfo = 'label+text+value'

fig.show()
fig_c = px.bar(day_wise, x="Date", y="Confirmed", color_discrete_sequence = [act])

fig_d = px.bar(day_wise, x="Date", y="Deaths", color_discrete_sequence = [dth])



fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.1,

                    subplot_titles=('Confirmed cases', 'Deaths reported'))



fig.add_trace(fig_c['data'][0], row=1, col=1)

fig.add_trace(fig_d['data'][0], row=1, col=2)



fig.update_layout(height=480)



# ===============================



fig_1 = px.line(day_wise, x="Date", y="Deaths / 100 Cases", color_discrete_sequence = [dth])

fig_2 = px.line(day_wise, x="Date", y="Recovered / 100 Cases", color_discrete_sequence = [rec])

fig_3 = px.line(day_wise, x="Date", y="Deaths / 100 Recovered", color_discrete_sequence = ['#333333'])



fig = make_subplots(rows=1, cols=3, shared_xaxes=False, 

                    subplot_titles=('Deaths / 100 Cases', 'Recovered / 100 Cases', 'Deaths / 100 Recovered'))



fig.add_trace(fig_1['data'][0], row=1, col=1)

fig.add_trace(fig_2['data'][0], row=1, col=2)

fig.add_trace(fig_3['data'][0], row=1, col=3)



fig.update_layout(height=480)



# ===================================



fig_c = px.bar(day_wise, x="Date", y="New cases", color_discrete_sequence = [act])

fig_d = px.bar(day_wise, x="Date", y="No. of countries", color_discrete_sequence = [dth])



fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.1,

                    subplot_titles=('No. of new cases everyday', 'No. of countries'))



fig.add_trace(fig_c['data'][0], row=1, col=1)

fig.add_trace(fig_d['data'][0], row=1, col=2)



fig.update_layout(height=480)
# new cases - cases per million people

temp = country_wise[country_wise['Population']>1000000]

fig_p = px.bar(temp.sort_values('Cases / Million People').tail(15), x="Cases / Million People", y="Country/Region", 

               text='Cases / Million People', orientation='h', color_discrete_sequence = ['#741938'])



# plot

fig = make_subplots(rows=1, cols=1, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.08,

                    subplot_titles=('Top Countries according to the number of Cases per Million People', 'Deaths reported'))



fig.add_trace(fig_p['data'][0], row=1, col=1)







fig.update_layout(height=500)



#plt.savefig('images/025pl.png')
# confirmed - deaths

fig_c = px.bar(country_wise.sort_values('Confirmed').tail(15), x="Confirmed", y="Country/Region", 

               text='Confirmed', orientation='h', color_discrete_sequence = [act])

fig_d = px.bar(country_wise.sort_values('Deaths').tail(15), x="Deaths", y="Country/Region", 

               text='Deaths', orientation='h', color_discrete_sequence = [dth])



# plot

fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.08,

                    subplot_titles=('Confirmed cases', 'Deaths reported'))



fig.add_trace(fig_c['data'][0], row=1, col=1)

fig.add_trace(fig_d['data'][0], row=1, col=2)

fig.update_layout(height=500)



#plt.savefig('026pl.png')
# recovered - active

fig_r = px.bar(country_wise.sort_values('Recovered').tail(15), x="Recovered", y="Country/Region", 

               text='Recovered', orientation='h', color_discrete_sequence = [rec])

fig_a = px.bar(country_wise.sort_values('Active').tail(15), x="Active", y="Country/Region", 

               text='Active', orientation='h', color_discrete_sequence = ['#333333'])



# plot

fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.08,

                    subplot_titles=('Recovered', 'Active cases'))



fig.add_trace(fig_r['data'][0], row=1, col=1)

fig.add_trace(fig_a['data'][0], row=1, col=2)

fig.update_layout(height=500)



#plt.savefig('027pl.png')
# death - recoverd / 100 cases

fig_dc = px.bar(country_wise.sort_values('Deaths / 100 Cases').tail(15), x="Deaths / 100 Cases", y="Country/Region", 

               text='Deaths / 100 Cases', orientation='h', color_discrete_sequence = ['#f38181'])

fig_rc = px.bar(country_wise.sort_values('Recovered / 100 Cases').tail(15), x="Recovered / 100 Cases", y="Country/Region", 

               text='Recovered / 100 Cases', orientation='h', color_discrete_sequence = ['#a3de83'])



# plot

fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.08,

                    subplot_titles=('Number of deaths per 100 cases', 'Number of recovered per 100 cases'))



fig.add_trace(fig_dc['data'][0], row=1, col=1)

fig.add_trace(fig_rc['data'][0], row=1, col=2)

fig.update_layout(height=500)



#plt.savefig('028pl.png')
# new cases - cases per million people

fig_nc = px.bar(country_wise.sort_values('New cases').tail(15), x="New cases", y="Country/Region", 

               text='New cases', orientation='h', color_discrete_sequence = ['#c61951'])

temp = country_wise[country_wise['Population']>1000000]

fig_p = px.bar(temp.sort_values('Cases / Million People').tail(15), x="Cases / Million People", y="Country/Region", 

               text='Cases / Million People', orientation='h', color_discrete_sequence = ['#741938'])



# plot

fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.08,

                    subplot_titles=('New cases', 

                                    'Cases / Million People'))



fig.add_trace(fig_nc['data'][0], row=1, col=1)

fig.add_trace(fig_p['data'][0], row=1, col=2)

fig.update_layout(height=500)



#plt.savefig('029pl.png')
# week change, percent increase

fig_wc = px.bar(country_wise.sort_values('1 week change').tail(15), x="1 week change", y="Country/Region", 

               text='1 week change', orientation='h', color_discrete_sequence = ['#004a7c'])

temp = country_wise[country_wise['Confirmed']>100]

fig_pi = px.bar(temp.sort_values('1 week % increase').tail(15), x="1 week % increase", y="Country/Region", 

               text='1 week % increase', orientation='h', color_discrete_sequence = ['#005691'], 

                hover_data=['Confirmed last week', 'Confirmed'])



# plot

fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.08,

                    subplot_titles=('1 week increase', '1 week % increase'))



fig.add_trace(fig_wc['data'][0], row=1, col=1)

fig.add_trace(fig_pi['data'][0], row=1, col=2)

fig.update_layout(height=500)

#plt.savefig('030pl.png')
fig = px.scatter(country_wise.sort_values('Deaths', ascending=False).iloc[:15, :], 

                 x='Confirmed', y='Deaths', color='Country/Region', size='Confirmed', height=700,

                 text='Country/Region', log_x=True, log_y=True, title='Deaths vs Confirmed (Scale is in log10)')

fig.update_traces(textposition='top center')

fig.update_layout(showlegend=False)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
fig = px.bar(full_grouped, x="Date", y="Confirmed", color='Country/Region', height=600,

             title='Confirmed', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()



# =========================================



fig = px.bar(full_grouped, x="Date", y="Deaths", color='Country/Region', height=600,

             title='Deaths', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()



# =========================================



fig = px.bar(full_grouped, x="Date", y="New cases", color='Country/Region', height=600,

             title='New cases', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()
fig = px.line(full_grouped, x="Date", y="Confirmed", color='Country/Region', height=600,

             title='Confirmed', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()



# =========================================



fig = px.line(full_grouped, x="Date", y="Deaths", color='Country/Region', height=600,

             title='Deaths', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()



# =========================================



fig = px.line(full_grouped, x="Date", y="New cases", color='Country/Region', height=600,

             title='New cases', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()
gt_100 = full_grouped[full_grouped['Confirmed']>100]['Country/Region'].unique()

temp = full_table[full_table['Country/Region'].isin(gt_100)]

temp = temp.groupby(['Country/Region', 'Date'])['Confirmed'].sum().reset_index()

temp = temp[temp['Confirmed']>100]

# print(temp.head())



min_date = temp.groupby('Country/Region')['Date'].min().reset_index()

min_date.columns = ['Country/Region', 'Min Date']

# print(min_date.head())



from_100th_case = pd.merge(temp, min_date, on='Country/Region')

from_100th_case['N days'] = (from_100th_case['Date'] - from_100th_case['Min Date']).dt.days

# print(from_100th_case.head())



fig = px.line(from_100th_case, x='N days', y='Confirmed', color='Country/Region', title='N days from 100 case', height=600)

fig.show()



# ===========================================================================



gt_1000 = full_grouped[full_grouped['Confirmed']>1000]['Country/Region'].unique()

temp = full_table[full_table['Country/Region'].isin(gt_1000)]

temp = temp.groupby(['Country/Region', 'Date'])['Confirmed'].sum().reset_index()

temp = temp[temp['Confirmed']>1000]

# print(temp.head())



min_date = temp.groupby('Country/Region')['Date'].min().reset_index()

min_date.columns = ['Country/Region', 'Min Date']

# print(min_date.head())



from_1000th_case = pd.merge(temp, min_date, on='Country/Region')

from_1000th_case['N days'] = (from_1000th_case['Date'] - from_1000th_case['Min Date']).dt.days

# print(from_1000th_case.head())



fig = px.line(from_1000th_case, x='N days', y='Confirmed', color='Country/Region', title='N days from 1000 case', height=600)

fig.show()



# ===========================================================================



gt_10000 = full_grouped[full_grouped['Confirmed']>10000]['Country/Region'].unique()

temp = full_table[full_table['Country/Region'].isin(gt_10000)]

temp = temp.groupby(['Country/Region', 'Date'])['Confirmed'].sum().reset_index()

temp = temp[temp['Confirmed']>10000]

# print(temp.head())



min_date = temp.groupby('Country/Region')['Date'].min().reset_index()

min_date.columns = ['Country/Region', 'Min Date']

# print(min_date.head())



from_10000th_case = pd.merge(temp, min_date, on='Country/Region')

from_10000th_case['N days'] = (from_10000th_case['Date'] - from_10000th_case['Min Date']).dt.days

# print(from_10000th_case.head())full_grouped



fig = px.line(from_10000th_case, x='N days', y='Confirmed', color='Country/Region', title='N days from 10000 case', height=600)

fig.show()
full_latest = full_table[full_table['Date'] == max(full_table['Date'])]

                         

fig = px.treemap(full_latest.sort_values(by='Confirmed', ascending=False).reset_index(drop=True), 

                 path=["Country/Region", "Province/State"], values="Confirmed", height=700,

                 title='Number of Confirmed Cases',

                 color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label+text+value'

fig.show()



fig = px.treemap(full_latest.sort_values(by='Deaths', ascending=False).reset_index(drop=True), 

                 path=["Country/Region", "Province/State"], values="Deaths", height=700,

                 title='Number of Deaths reported',

                 color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label+text+value'

#fig.show()
temp = full_grouped[full_grouped['New cases']>0].sort_values('Country/Region', ascending=False)

fig = px.scatter(temp, x='Date', y='Country/Region', size='New cases', color='New cases', height=3000, 

           color_continuous_scale=px.colors.sequential.Viridis)

fig.update_layout(yaxis = dict(dtick = 1))

fig.update(layout_coloraxis_showscale=False)

#fig.show()
# first date

# ==========

first_date = full_table[full_table['Confirmed']>0]

first_date = first_date.groupby('Country/Region')['Date'].agg(['min']).reset_index()

# first_date.head()



# last date

# =========

last_date = full_table.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']

last_date = last_date.sum().diff().reset_index()



mask = last_date['Country/Region'] != last_date['Country/Region'].shift(1)

last_date.loc[mask, 'Confirmed'] = np.nan

last_date.loc[mask, 'Deaths'] = np.nan

last_date.loc[mask, 'Recovered'] = np.nan



last_date = last_date[last_date['Confirmed']>0]

last_date = last_date.groupby('Country/Region')['Date'].agg(['max']).reset_index()

# last_date.head()



# first_last

# ==========

first_last = pd.concat([first_date, last_date[['max']]], axis=1)



# added 1 more day, which will show the next day as the day on which last case appeared

first_last['max'] = first_last['max'] + timedelta(days=1)



# no. of days

first_last['Days'] = first_last['max'] - first_last['min']



# task column as country

first_last['Task'] = first_last['Country/Region']



# rename columns

first_last.columns = ['Country/Region', 'Start', 'Finish', 'Days', 'Task']



# sort by no. of days

first_last = first_last.sort_values('Days')

# first_last.head()



# visualization

# =============



# produce random colors

clr = ["#"+''.join([random.choice('0123456789ABC') for j in range(6)]) for i in range(len(first_last))]



# plot

fig = ff.create_gantt(first_last, index_col='Country/Region', colors=clr, show_colorbar=False, 

                      bar_width=0.2, showgrid_x=True, showgrid_y=True, height=2500)

#fig.show()
temp = full_table.groupby(['Date', 'Country/Region'])['Confirmed'].sum()

temp = temp.reset_index().sort_values(by=['Date', 'Country/Region'])

temp = temp[temp['Country/Region'].isin(gt_1000)]



plt.style.use('seaborn')

g = sns.FacetGrid(temp, col="Country/Region", hue="Country/Region", sharey=False, col_wrap=5)

g = g.map(plt.plot, "Date", "Confirmed")

g.set_xticklabels(rotation=90)

plt.show()
temp = full_table.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']

temp = temp.sum().diff().reset_index()

temp = temp[temp['Country/Region'].isin(gt_1000)]



mask = temp['Country/Region'] != temp['Country/Region'].shift(1)



temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan

temp.loc[mask, 'Recovered'] = np.nan



# plt.style.use('seaborn')

# g = sns.FacetGrid(temp, col="Country/Region", hue="Country/Region",  sharey=False, col_wrap=5)

# g = g.map(sns.lineplot, "Date", "Confirmed")

# g.set_xticklabels(rotation=90)

# plt.show()
epidemics = pd.DataFrame({

    'epidemic' : ['COVID-19', 'SARS', 'EBOLA', 'MERS', 'H1N1'],

    'start_year' : [2019, 2003, 2014, 2012, 2009],

    'end_year' : [2020, 2004, 2016, 2017, 2010],

    'confirmed' : [full_latest['Confirmed'].sum(), 8096, 28646, 2494, 6724149],

    'deaths' : [full_latest['Deaths'].sum(), 774, 11323, 858, 19654]

})



epidemics['mortality'] = round((epidemics['deaths']/epidemics['confirmed'])*100, 2)



epidemics.head()
!yes Y | conda install -c plotly plotly-orca



import plotly.io as pio

import plotly as plotly

plotly.io.orca.config.executable = '/path/to/orca'

pio.orca.config.use_xvfb = True

pio.orca.config.save()
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