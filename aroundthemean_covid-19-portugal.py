import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.stats.api as sms



from datetime import datetime, timedelta



%matplotlib inline

sns.set()
# Load Complete Dataset

df_covid_complete = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")



# Filter only portuguese data

df_pt_covid = df_covid_complete[df_covid_complete['Country/Region']=='Portugal']



# Clean Portuguese data - Drop Redundant Columns

df_pt_covid = df_pt_covid.drop(columns=['Province/State','Country/Region']).reset_index()

# Clean Portuguese data - Drop Rows Before First Confirmed Case

df_pt_covid = df_pt_covid.iloc[np.max(df_pt_covid[df_pt_covid['Confirmed']==0].index.values):].reset_index()
# Plot Confirmed cases

plt.figure(figsize=(15,7))

plt.stem(df_pt_covid.Date.values,df_pt_covid.Confirmed.values)

plt.title('Evolution of the number of Confirmed cases of COVID-19 Infection in Portugal', fontweight='bold')

plt.ylabel('Number of confirmed cases')

plt.xlabel('Date')

plt.xticks(rotation=45);
#Find the multiplicative value between days R0

confirmed = df_pt_covid.Confirmed.values



R = []

for i in range(2,len(confirmed)-1):

    R.append(confirmed[i+1]/confirmed[i])



R = np.array(R)



plt.figure(figsize=(15,7))

plt.plot(df_pt_covid.Date.values[3:],R)

plt.title('Evolution of the propagation rate of COVID-19 in Portugal', fontweight='bold')

plt.ylabel('Propagation Rate')

plt.xlabel('Date')

plt.xticks(rotation=45)

R0 = np.mean(R)

print('Average propagation rate: \t'+str(R0))
# Fit an exponential model to the data, based on R0

days_ahead = 3



fit_confirmed = np.power(R0,range(0,len(confirmed)))

pred_confirmed = np.power(R0,range(0,len(confirmed)+days_ahead))



dates = list(df_pt_covid.Date.values)

for _ in range(days_ahead):

    date = dates[-1]

    date = datetime.strptime(date,'%m/%d/%y') + timedelta(days=1)

    date = date.strftime('%m/%d/%y')

    dates.append(date)



plt.figure(figsize=(15,7))

plt.stem(df_pt_covid.Date.values,df_pt_covid.Confirmed.values)

plt.plot(dates,pred_confirmed,'r')

plt.plot(df_pt_covid.Date.values,fit_confirmed,'g')

plt.title('Evolution of the number of Confirmed cases of COVID-19 Infection in Portugal', fontweight='bold')

plt.ylabel('Number of confirmed cases')

plt.xlabel('Date')

plt.xticks(rotation=45);
# Plot with confidence interval margins (90%)

conf_int = sms.DescrStatsW(R).tconfint_mean(alpha=0.1)



pred_confirmed_lower_bound = np.power(conf_int[0],range(0,len(confirmed)+days_ahead))

pred_confirmed_upper_bound = np.power(conf_int[1],range(0,len(confirmed)+days_ahead))

# pred_confirmed = np.power(R0,range(0,len(confirmed)+days_ahead))



plt.figure(figsize=(15,7))

plt.plot(dates,pred_confirmed_lower_bound, 'b', linewidth=3)

plt.plot(dates,pred_confirmed_upper_bound,'b', linewidth=3)

plt.plot(dates,pred_confirmed,'b', linewidth=3)

plt.fill_between(dates, pred_confirmed_lower_bound, pred_confirmed_upper_bound,

                 color='b', alpha=0.2)

plt.title('Evolution of the number of Confirmed cases of COVID-19 Infection in Portugal', fontweight='bold')

plt.ylabel('Number of confirmed cases')

plt.xlabel('Date')

plt.xticks(rotation=45);