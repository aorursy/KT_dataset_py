import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
from matplotlib import style
style.use ("fivethirtyeight")
data=pd.read_csv('../input/state-unemployment-rate-change-march-to-april-2020/States_with_statistically_significant_unemployment_rate_changes_from_March_2020_to_April_2020_season.csv')
df=pd.DataFrame (data)
df.sample(10)
State = df ['State']
march = df ['March 2020 (Rate)']
april = df ['April 2020 (p-Rate)']
y_pos = np.arange(len(State))
plt.bar(y_pos, march, align='center', alpha=.80)
plt.bar(y_pos, april, align='edge', alpha=0.75)
plt.xticks(y_pos, State)
plt.ylabel('\nUnemployment Rate (Percentage)\n')
plt.title('\nUnemployment Rate In Different US States From March to April in 2020\n')
plt.grid (True, color='gray')
months=['March 2020', 'April 2020']
plt.legend(months)
plt.rcParams['figure.figsize'] = [20, 10]
plt.show()
data=pd.read_csv('../input/state-employment-changes-march-to-april-2020/States_with_statistically_significant_employment_changes_from_March_2020_to_April_2020_seasonally_ad.csv')
df=pd.DataFrame (data)
df.sample(10)
State = df ['State']
march = df ['March 2020']
april = df ['April 2020 (p)']
y_pos = np.arange(len(State))
plt.bar(y_pos, march, align='center', color='#E0115F', alpha=.80)
plt.bar(y_pos, april, align='center', color='#002147',alpha=0.75)
plt.xticks(y_pos, State)
plt.ylabel('\nEmployed Population\n')
plt.title('\nEmployment Change In Different US States From March to April in 2020\n')
plt.grid (False, color='grey')
months=['March 2020', 'April 2020']
plt.legend(months)
plt.rcParams['figure.figsize'] = [20, 10]
plt.show()
data=pd.read_csv('../input/local-unemployment-report-district-of-columbia/Local Unemployment Report District of Columbia.csv')
df=pd.DataFrame (data)
df.sample(10)
months = df ['Period'].tail(12)
unemployed = df ['unemployment'].tail(12)
plt.plot (months,unemployed, label='Unemployed Population', color='#ffd700')
plt.title ("\nUnemployment Statistics In District of Columbia From May 2019 to April 2020\n", color='k')
plt.xlabel ("\nLast 12 Months\n", color= 'k')
plt.ylabel ("\nUnemployed Population\n", color= 'k')
plt.legend ()
plt.grid (True, color='#6E8B3D')
plt.rcParams['figure.figsize'] = [20, 10]
plt.show()
data=pd.read_csv('../input/state-unemployment-rates-april-2020/States_with_new_series_high_unemployment_rates_April_2020_seasonally_adjusted_-_Sheet1.csv')
df=pd.DataFrame (data)
df.sample(10)
States = df ['State']
rate = df ['Rate(p)']
y_pos = np.arange(len(States))
plt.bar(y_pos, rate, color='#8B008B', align='center', alpha=.70)
plt.xticks(y_pos, States)
plt.ylabel('\nUnemployment Rate (Percentage)\n')
plt.title('\nUnemployment Rate In Different US States\n')
plt.grid (True, color='gray')
plt.rcParams['figure.figsize'] = [25, 10]
plt.show()
data=pd.read_csv('../input/who-covid19-us-data/WHO-COVID-19-US-Data.csv')
df=pd.DataFrame (data)
df.sample(10)
date = df ['Date_reported'].tail(62)
covid_positive_cumulative = df ['Cumulative_cases'].tail(62)
deaths_cumulative = df ['Cumulative_deaths'].tail(62)
y_pos = np.arange(len(date))
plt.bar(y_pos, covid_positive_cumulative, align='center', color='#00008B', alpha=.80),
plt.bar(y_pos, deaths_cumulative, align='center', color='#A40000',alpha=0.75),
plt.xticks(y_pos)
plt.ylabel('\nCumulative Number\n')
plt.title('\nSpread of COVID19 in US from 27.03.2020 to 27.05.2020\n')
plt.grid (True, color='grey')
leg=['Cumulative COVID Positive Cases', 'Cumulative Deaths']
plt.legend(leg)
plt.rcParams['figure.figsize'] = [25, 10]
plt.show()
data=pd.read_csv('../input/new-york-mobility-report/New York Mobility Report.csv')
df=pd.DataFrame (data)
df.sample (10)
date = df ['Date '].tail(91)
retail = df ['retail_and_recreation_percent_change_from_baseline '].tail(91)
grocery = df ['grocery_and_pharmacy_percent_change_from_baseline '].tail(91)
parks = df ['parks_percent_change_from_baseline '].tail(91)
transit = df ['transit_stations_percent_change_from_baseline '].tail(91)
y_pos = np.arange(len(date))
plt.bar(y_pos, parks, align='center', width=0.5, color='#006B3C', alpha=.80),
plt.bar(y_pos, grocery, align='center', color='#7B3F00',alpha=0.75),
plt.xticks(y_pos)
plt.ylabel('\nRate of Change\n')
plt.title('\nChange in Activity of New York People in 25.02.2020 to 25.05.2020\n')
plt.grid (True, color='grey')
leg=['Parks', 'Grocery Shops']
plt.legend(leg)
plt.rcParams['figure.figsize'] = [30, 15]
plt.show()
