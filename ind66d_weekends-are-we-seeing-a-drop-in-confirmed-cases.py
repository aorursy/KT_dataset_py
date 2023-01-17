import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/covid-latest/covid_19_clean_complete.csv')
fig, ax = plt.subplots(1,2,figsize=(30,8))



X = df[(df['Country/Region']=='Germany')][df.Date>='2020-07-01'][df.Date<'2020-08-01']



ax[0].bar(range(X.shape[0]),X.Confirmed.diff())



ax[0].set_xticks(np.arange(X.shape[0]))

ax[0].set_xticklabels(pd.to_datetime(X.Date).dt.date, rotation=90)

ax[0].set_title('New Confirmed Cases - Germany')



X = df[(df['Country/Region']=='Italy')][df.Date>='2020-07-01'][df.Date<'2020-08-01']



ax[1].bar(range(X.shape[0]),X.Confirmed.diff())



ax[1].set_xticks(np.arange(X.shape[0]))

ax[1].set_xticklabels(pd.to_datetime(X.Date).dt.date, rotation=90)

ax[1].set_title('New Confirmed Cases - Italy')



plt.show()
df.info()
df = df[['Province/State','Country/Region','Date','Confirmed']]
for states_country in df[~df['Province/State'].isnull()]['Country/Region'].unique():

    X = df[df['Country/Region']==states_country].groupby(['Country/Region','Date']).sum().reset_index()

    df.drop(df[df['Country/Region']==states_country].index, inplace=True)

    df = df.append(X).reset_index(drop=True)



df.drop(['Province/State'], axis=1, inplace=True)
df.sort_values(['Country/Region','Date'], ascending=[True, True], inplace=True)

df.reset_index(drop=True, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

df['Weekday'] = df['Date'].dt.day_name()

df.head()
country_row_count = df[df['Country/Region']==df['Country/Region'].unique()[0]].shape[0]

print(country_row_count, ((df.groupby('Country/Region')['Country/Region'].count()-country_row_count)==0).all())

weeknum = [(x//7)+1 for x in range(2,country_row_count+2)]*len(df['Country/Region'].unique())

df['Weeknum'] = weeknum

df
df['New_Confirmed'] = df.groupby(['Country/Region']).Confirmed.diff()

df['New_Confirmed'].fillna(df['Confirmed'],inplace=True)
import requests

from bs4 import BeautifulSoup



URL = "https://en.wikipedia.org/wiki/Workweek_and_weekend"



res = requests.get(URL).text

soup = BeautifulSoup(res,'lxml')

req = soup.find('table', class_='wikitable').find_all('tr')[1::1]



final = []



for ln in req:

    if len(ln)>5:

        final.append([str(list(ln)[1]).replace('<td>','').replace('</td>',''), str(list(ln)[5]).replace('<td>','').replace('</td>','').split()[0]])



sc = pd.DataFrame(final,columns=['Country','Workdays'])

sc['Free_Sunday'] = sc.Workdays.apply(lambda x: True if (((('Friday' in x) or ('Saturday' in x)) and ('Sunday' not in x)) or (('Monday' in x) and ('Sunday' not in x))) else False)

No_Sunday_Countries = sc[~sc.Free_Sunday].Country.to_list()



print('\033[1mList of countries listed our dataset but not on the wikipedia table:\n\033[0m', [cn for cn in df['Country/Region'].unique() if cn not in sc['Country'].unique()])

print('\n\033[1mList of countries with working Sundays that are in our dataset:\n\033[0m', [cn for cn in No_Sunday_Countries if cn in df['Country/Region'].unique()])



df.drop(df[df['Country/Region'].isin(No_Sunday_Countries)].index, inplace=True)

df.reset_index(drop=True, inplace=True)
WeeklyMean = df.groupby(['Country/Region','Weeknum']).New_Confirmed.mean()

WeeklyMeanGlobal = df.groupby(['Weeknum']).New_Confirmed.mean()



X = df[(df['Weekday'].isin(['Sunday','Monday']))]

X['Confirmed_diff'] = X.New_Confirmed.diff()



Xglobal = X.groupby(['Date','Weeknum','Weekday']).New_Confirmed.mean().reset_index()

Xglobal['Confirmed_diff'] = Xglobal.New_Confirmed.diff()
fig, ax = plt.subplots(figsize=(15,8))



c_name='Global'



Y = Xglobal[Xglobal['Weekday']=='Monday']



ax.bar(range(Y.shape[0]),WeeklyMeanGlobal.to_list()[:Y.shape[0]],color='peru', label='Confirmed Weekly (per day) Avg.')

ax.bar(range(Y.shape[0]),Y.New_Confirmed, alpha=0.9,label="Confirmed Next Monday")

ax.bar(range(Y.shape[0]),Y.Confirmed_diff, alpha=0.55,label='Monday/Sunday Difference',color='red',width=0.33)



ax.set_xticks(np.arange(Y.shape[0]))

ax.set_xticklabels(Y.Date.dt.date, rotation=90)

ax.set_title(c_name)

ax.legend()
from bokeh.io import show ,output_notebook

from bokeh.plotting import figure

from bokeh.models import Panel, Tabs

from datetime import timedelta

output_notebook()



tabs = []

p = []



for c_name in np.sort(abs(X.groupby(['Country/Region']).Confirmed_diff.mean()).sort_values().index.to_list()):

    Y = X[(X['Country/Region']==c_name)&(X['Weekday']=='Monday')]

    p.append(figure(plot_width=920, plot_height=480,x_axis_type='datetime',min_border=0))

    p[-1].xaxis.major_label_orientation = "vertical"

    p[-1].vbar(x=Y.Date.dt.date, top=WeeklyMean[c_name].to_list()[:Y.shape[0]], width=timedelta(days=5),color="royalblue",legend_label='Confirmed Weekly per Day Avg.')

    p[-1].vbar(x=Y.Date.dt.date, top=Y.New_Confirmed, width=timedelta(days=5),color="peru",fill_alpha=0.6,legend_label='Confirmed Next Monday')

    p[-1].vbar(x=Y.Date.dt.date, top=Y.Confirmed_diff, width=timedelta(days=1),color="red",fill_alpha=0.25,legend_label='Monday/Sunday Difference')

    p[-1].legend.location = "top_left"

    p[-1].legend.click_policy="hide"

    tabs.append(Panel(child=p[-1], title=c_name))



show(Tabs(tabs=tabs))