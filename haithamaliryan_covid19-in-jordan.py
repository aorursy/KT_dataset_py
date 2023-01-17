import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
%matplotlib inline 
#extract our data
confirmed= pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths= pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recoveries= pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
deaths.head()
recoveries.head()

confirmed.head()
confirmed.describe()
last_date=confirmed.columns[-1]
last_confirmed_sum=confirmed[last_date].sum()
last_recoveries_sum=recoveries[last_date].sum()
last_deaths_sum=deaths[last_date].sum()
print("No of cassess over the world ={} on date {}".format(last_confirmed_sum,last_date))
print("No of Recovered cassess over the world ={} on date {}".format(last_recoveries_sum,last_date))
print("No of Deathes cassess over the world ={} on date {}".format(last_deaths_sum,last_date))
col=confirmed.columns
col_dates=col[4:]
col_dates
#Jordan Data
confirmed_df_jordan=confirmed[confirmed['Country/Region']=="Jordan"]
deaths_df_jordan=deaths[deaths['Country/Region']=="Jordan"]
recoveries_df_jordan=recoveries[recoveries['Country/Region']=="Jordan"]
#DEtermin row no for jordan
jordan_row=confirmed_df_jordan.index.values[0]
#Determin first Date 
first_date=""
for i in col_dates:
    if (confirmed_df_jordan.loc[jordan_row,i])!=0 :
        first_date=i
        break
first_date
#dtermin col no for first patient
col.get_loc(first_date)
jordan_casses=confirmed_df_jordan.iloc[0,45:]
jordan_casses

plt.figure(figsize=(16, 9))
plt.plot(np.array(jordan_casses))
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 3/3/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
plt.figure(figsize=(16, 9))
plt.plot(np.log10(list(jordan_casses)))
plt.title('# of Coronavirus Cases Over Time Log Scale', size=30)
plt.xlabel('Days Since 3/3/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
total_deaths_jordan = deaths_df_jordan.iloc[0,45:]
plt.figure(figsize=(16, 9))
plt.plot(np.array(total_deaths_jordan))
plt.title('# of Coronavirus Deaths Over Time', size=30)
plt.xlabel('Days Since 3/3/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
recoveries_df_jordan=recoveries_df_jordan.iloc[0,45:]
plt.figure(figsize=(16, 9))
plt.plot(np.array(recoveries_df_jordan))
plt.title('# of Coronavirus recoveries Over Time', size=30)
plt.xlabel('Days Since 3/3/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
mortality_rate=total_deaths_jordan/jordan_casses

mean_mortality_rate = np.mean(mortality_rate)
plt.figure(figsize=(16, 9))
plt.plot(list(mortality_rate), color='orange')
plt.axhline(y = mean_mortality_rate,linestyle='--', color='black')
plt.title('Mortality Rate of Coronavirus Over Time', size=30)
plt.legend(['mortality rate', 'y='+str(mean_mortality_rate)], prop={'size': 20})
plt.xlabel('Days Since 03/03/2020', size=30)
plt.ylabel('Mortality Rate', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

#jordan
mortality_rate_all=total_deaths_jordan[-1]/jordan_casses[-1]*100
#global
global_mortality_rate = deaths.iloc[:,-1].sum()/confirmed.iloc[:,-1].sum()*100
print("based on {} global mortality rate is {} and in jordan is {}".format(last_date,round(global_mortality_rate,2),round(mortality_rate_all,2)))
recovery_rate=recoveries_df_jordan/jordan_casses
#jordan
recovery_rate_all=recoveries_df_jordan[-1]/jordan_casses[-1]*100
#global
global_recovery_rate_all=recoveries.iloc[:,-1].sum()/confirmed.iloc[:,-1].sum()*100
print("based on {} global recoveries rate is {} and in jordan is {}".format(last_date,round(global_recovery_rate_all,2),round(recovery_rate_all,2)))
mean_recovery_rate = np.mean(recovery_rate)
plt.figure(figsize=(16, 9))
plt.plot(list(recovery_rate), color='orange')
plt.axhline(y = mean_recovery_rate,linestyle='--', color='black')
plt.title('mean_recovery_rate of Coronavirus Over Time', size=30)
plt.legend(['mean_recovery_rate', 'y='+str(mean_recovery_rate)], prop={'size': 20})
plt.xlabel('Days Since 03/03/2020', size=30)
plt.ylabel('recovery_rate', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
def daily_increase2(data):
    d = []
    d.append(data[0])
    for i in range(1,len(data)):
        d.append(data[i]-data[i-1])
    return d

daily_increase_casses=daily_increase2(jordan_casses)
daily_increase_casses
plt.figure(figsize=(16, 9))
plt.plot(daily_increase_casses)
plt.title('daily_increase_casses of Coronavirus Over Time', size=30)
plt.xlabel('Days Since 03/03/2020', size=30)
plt.ylabel('daily_increase_casses', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
jordan_casses['3/21/20']=99
daily_increase_casses=daily_increase2(jordan_casses)
plt.figure(figsize=(16, 9))
plt.plot(daily_increase_casses)
plt.title('daily_increase_casses of Coronavirus Over Time', size=30)
plt.xlabel('Days Since 03/03/2020', size=30)
plt.ylabel('daily_increase_casses', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
jordan_population=10618522
round(jordan_casses[-1]/jordan_population*100,4)
