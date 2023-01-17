import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from dateutil.parser import parse

import re
data = pd.read_csv("../input/attacks.csv", encoding = 'ISO-8859-1')

data.head()
#data['Age'].fillna(0)

#data.Age.value_counts()

data['isAge'] = data['Age'].apply(lambda x: str(x).isdigit())

data = data[data['isAge'] == True]

data['Age'] = data['Age'].astype(int)
#Plot the distribution of ages

fig = sns.distplot(data['Age'])

ax = fig.axes

ax.set_xlim(0,)


data.loc[data['Fatal (Y/N)'] == " N", 'Fatal (Y/N)'] = "N"

data = data[data['Fatal (Y/N)'] != "UNKNOWN"]

data['Fatal (Y/N)'].value_counts()
fat_dfs = data.groupby('Fatal (Y/N)')

count = 0

for df in fat_dfs:

    df = df[1]

    #print(df['Fatal (Y/N)'].unique())

    if count == 0:

        lab = 'Non-fatal'

    else:

        lab = 'Fatal'

    print('Average age in', lab.lower(), 'incidents: ' , df['Age'].mean())

    sns.distplot(df['Age'], label = lab)

    plt.legend()

    count = count + 1



#Seems like the average age in fatal incidents is slightly higher than in non-fatal incidents
data['temp_dates'] = data['Date'].apply(lambda x: x.split('-'))

data['temp_dates']

#temp_dates[1][1]



#day = [x[0] for x in temp_dates]

#month = [x[1] for x in temp_dates]

#year = [x[2] for x in temp_dates]
def find_dates(date):

    try: 

        m = re.search('(\d\d\d\d\.\d\d\.\d\d)', date)

        return m.group(0)

    except:

        return 0
def convert_todate(date):

    try:

        return pd.to_datetime(date)

    except:

        return 0
data['temp'] = data['Case Number'].apply(find_dates)

data = data[(data['temp'] != '0')]

data = data[(data['temp'] != 0)]
data['date'] = data['temp'].apply(lambda x: convert_todate(x))

del data['temp']

data = data[data['date'] != 0]

#data['date'] = pd.to_datetime(data['temp'])

#data[data['Case Number'] == '1703.03.26']
data['month'] = data['date'].apply(lambda x: x.month)

data['year'] = data['date'].apply(lambda x: x.year)

data['day'] = data['date'].apply(lambda x: x.dayofweek)
#Group data into various time intervals

year_inc = data.groupby('year')

month_inc = data.groupby('month')

day_inc = data.groupby('day')

time_dfs = [year_inc, month_inc, day_inc]
#DAILY ANALYSIS
daily_incidents = []

daily_age = []

daily_fatalities = []

for day in day_inc:

    day = day[1]

    daily_incidents.append(day['Case Number'].count())

    daily_age.append(day['Age'].mean())

    daily_fatalities.append(day[day['Fatal (Y/N)'] == 'Y']['Fatal (Y/N)'].count())

    print(day['day'].unique(), day['Age'].mean())

pct_fatal = [x/y for x,y in zip(daily_fatalities,daily_incidents)]

day_labs = ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday']

day_df = pd.DataFrame([day_labs,daily_incidents, daily_age, daily_fatalities, pct_fatal]).transpose()

day_df.columns = ['day','incidents','age','fatalities', 'pct_fatal']

sns.barplot(x = 'day', y ='incidents', data= day_df)
sns.barplot(x = 'day', y ='pct_fatal', data= day_df)
#MONTHLY ANALYSIS
monthly_incidents = []

monthly_age = []

monthly_fatalities = []

for mo in month_inc:

    mo = mo[1]

    monthly_incidents.append(mo['Case Number'].count())

    monthly_age.append(mo['Age'].mean())

    monthly_fatalities.append(mo[mo['Fatal (Y/N)'] == 'Y']['Fatal (Y/N)'].count())

    

mo_pct_fatal = [x/y for x,y in zip(monthly_fatalities,monthly_incidents)]

month_labs = ['January', 'February', 'March','April','May','June','July', 'August', 'September','October','November','December']

month_df = pd.DataFrame([month_labs,monthly_incidents, monthly_age, monthly_fatalities, mo_pct_fatal]).transpose()

month_df.columns = ['month','incidents','age','fatalities', 'pct_fatal']

sns.barplot(x = 'month', y ='incidents', data= month_df)
#Monthly percentage of incidents that were fatal

sns.barplot(x = 'month', y ='pct_fatal', data= month_df)
yearly_incidents = []

yearly_age = []

yearly_fatalities = []

year = []

for yr in year_inc:

    yr = yr[1]

    if (yr['year'].unique() == 1703):

        print(yr['year'].unique())

        continue

    year.append(str(yr['year'].unique()).replace("[","").replace("]","").replace("'",""))

    yearly_incidents.append(yr['Case Number'].count())

    yearly_age.append(yr['Age'].mean())

    yearly_fatalities.append(yr[yr['Fatal (Y/N)'] == 'Y']['Fatal (Y/N)'].count())
#Plot of the number of incidents by year

yr_pct_fatal = [x/y for x,y in zip(yearly_fatalities,yearly_incidents)]

year_df = pd.DataFrame([year,yearly_incidents, yearly_age, yearly_fatalities, yr_pct_fatal]).transpose()

year_df.columns = ['year','incidents','age','fatalities', 'pct_fatal']

year_plot = sns.barplot(x = 'year', y ='incidents', data= year_df)



for item in year_plot.get_xticklabels():

    item.set_rotation(90)



#Display every 10th year's label

for ind, label in enumerate(year_plot.get_xticklabels()):

    if ind % 10 == 0:  

        label.set_visible(True)

    else:

        label.set_visible(False)
#Percentage of incidents that were fatal

year_fatal_inc = sns.barplot(x = 'year', y ='pct_fatal', data= year_df)



for item in year_fatal_inc.get_xticklabels():

    item.set_rotation(90)



#Display every 10th year's label

for ind, label in enumerate(year_fatal_inc.get_xticklabels()):

    if ind % 10 == 0:  

        label.set_visible(True)

    else:

        label.set_visible(False)