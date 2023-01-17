import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import seaborn as sns

%matplotlib inline
df = pd.read_csv('/kaggle/input/uncover/UNCOVER/geotab/border-wait-times-at-us-canada-border.csv')
df.shape
df.columns
df.info()
df.isnull().sum()
df.head(3)
df.borderid.unique(),df.canadaport.unique()
df.utc_date.value_counts()
# not needed columns

# borderid, version

#
df.americaport.nunique(),df.localdate.nunique()
df.tripdirection.nunique()
df.daytype.unique()
df.can_iso_3166_2.nunique()
df.daytype.value_counts().plot(kind='pie')
ax = df.groupby('daytype')['averageduration'].agg(np.mean)

ax
# ax = df.groupby('tripdirection','daytype')['averageduration'].agg(np.mean)

# ax

sns.countplot(x = "tripdirection", hue = "daytype", data = df).set_title('Total number of people travelled')
def fun(time):

    if time < 3:

        return "Below3_Hr"

    elif 3 <= time < 5:

        return "Between_3_to_5Hr"

    else:

        return "Morethan_5Hr"



df['timetaken'] = df.averageduration.apply(lambda x: fun(x))
ax.plot.bar(title='Average time duration')
df_avg4 = df[df['averageduration'] >5]

df_avg4.shape
ax = df_avg4.groupby('daytype')['averageduration'].sum()

ax
ax = df.groupby(['daytype'])['timetaken'].value_counts()

ax
ax.unstack().plot(kind='bar')
freq_df = df.groupby(['timetaken'])['daytype'].value_counts().unstack(0)

plt.figure(figsize=(10,10))

pct_df = freq_df.divide(freq_df.sum(axis=1), axis=0)

pct_df = pct_df.mul(100)

pct_df.plot(kind="bar", stacked=False,figsize=(6,6),rot=0,title='Time taken at weekdays & weekends')
#df = df.sample(100)

df.daytype.nunique()
df.localhour.unique()
def hour(x):

    if 0<= x < 5:

        return 'Midnight'

    elif 5<= x < 12:

        return 'Morning'

    elif 12 <= x < 17:

        return 'Afternoon'

    elif 17 <=x <= 24:

        return 'Evening'

df['whattime_day'] = df.utc_hour.apply(lambda x: hour(x))
df.head(3)
df.shape
sr = df.whattime_day.value_counts()

sr = sr.to_frame()

sr = sr.reset_index()
#df.whattime_day.value_counts().plot('pie')

sr = sr.rename(columns = {'index'

                          :'names'})

sr

#plt.pie(df['whattime_day'].value_counts(),labels=sr.names.values)
sr = sr.rename(columns = {'index'

                          :'names'})
sr.names.values
#lis = df.whattime_day.unique()

labels = sr.names.values

sizes = df.whattime_day.value_counts(ascending=False)

explode = (0.1, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
df.utc_date[1200]
def utc_month(x):

    if x.split('-')[1] == '01':

        return 'Jan'

    elif x.split('-')[1] == '02':

        return 'Feb'

    elif x.split('-')[1] == '03':

        return 'March'

    elif x.split('-')[1] == '04':

        return 'April'

    elif x.split('-')[1] == '05':

        return 'May'

    elif x.split('-')[1] == '12':

        return 'Dec'

    else:

        y = x.split('-')[1]

        return y
df['month'] = df.localdate.apply(lambda x: utc_month(x))
df.month.value_counts()
sr = df.month.value_counts()

sr = sr.to_frame()

sr = sr.reset_index()

sr = sr.rename(columns = {'index'

                          :'names'})
labels = sr.names.values

sizes = df.month.value_counts(ascending=False)

explode = (0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
df.groupby(['tripdirection','month'])['month'].value_counts()
freq_df = df.groupby(['tripdirection'])['month'].value_counts().unstack()

plt.figure(figsize=(10,10))

my_colors = 'rgbymc'

palette = {"Jan":'C0',"Feb":"C1","March":"C2","April":"C3"}

pct_df = freq_df.divide(freq_df.sum(axis=1), axis=0)

pct_df = pct_df.mul(100)

pct_df.plot(kind="bar", stacked=False,figsize=(8,5),rot=0,title='People travelled across')
df.timetaken.unique()
df_port = df[df['timetaken'] == 'Morethan_5Hr']
df_port.shape
df_port['canadaport'].value_counts().head(10).plot(kind='bar',)
freq_df = df_port.groupby(['canadaport','month'])['timetaken'].value_counts().head(10).unstack(0)

plt.figure(figsize=(10,10))

my_colors = 'rgbymc'

#palette = {"Jan":'C0',"Feb":"C1","March":"C2","April":"C3"}

pct_df = freq_df.divide(freq_df.sum(axis=1), axis=0)

pct_df = pct_df.mul(100)

pct_df.plot(kind="bar", stacked=False,figsize=(8,7),rot=0,title='Various airports in different months')
df_port['americaport'].value_counts().head(10).plot(kind='bar')
freq_df = df_port.groupby(['americaport','month'])['timetaken'].value_counts().head(10).unstack(0)

plt.figure(figsize=(10,10))

my_colors = 'rgbymc'

#palette = {"Jan":'C0',"Feb":"C1","March":"C2","April":"C3"}

pct_df = freq_df.divide(freq_df.sum(axis=1), axis=0)

pct_df = pct_df.mul(100)

pct_df.plot(kind="bar", stacked=False,figsize=(8,7),rot=0,title='Various airports in US in different months')
df.utc_date[0]
def utc_week(x):

    a = x.split('-')[2]

    a = int(a)

    if 1<= a <=7:

        return 'Week1'

    elif 7 <a<=14:

        return 'Week2'

    elif 14 <a<=21:

        return 'Week3'

    else:

        return 'Week4'



df['week'] = df.utc_date.apply(lambda x: utc_week(x))
df.week.value_counts()
df_usport = df[df['timetaken'] == 'Morethan_5Hr']
freq_df = df_usport.groupby(['week','month'])['timetaken'].value_counts().unstack(0)

plt.figure(figsize=(10,10))

my_colors = 'rgbymc'

#palette = {"Jan":'C0',"Feb":"C1","March":"C2","April":"C3"}

pct_df = freq_df.divide(freq_df.sum(axis=1), axis=0)

pct_df = pct_df.mul(100)

pct_df.plot(kind="bar", stacked=False,figsize=(9,8),rot=0,title='Time taken in various weeks in diff months')
df.columns
# Important Insights we get from the above analysis:

# 1.Number of people who travelled from US to Canada & Canada to US are approximately same as it might be all the people who went on some work returned home.

# 2.Average time taken in weekdays is 7 and weekends is 8 as most people might have holidays on the weekends.

# 3.In weekdays as well as weekends for more than 55% people time taken is more than 5hrs.

# 4.The percentage of people who travelled in the evenings are around 38% followed by afternoon 25%.

# 5.Number of people travelled got decreased a bit from Jan to Feb but from Feb to March it covered up and in April its very less that is 3.1% that might be due to lockdown in countries

# 6.Top 5 airports where the average time taken in more than 5hrs are Windsor-Ambasador bridge, Samia, Fort Erie, Queenston, St.Bernard & that of America are Port Huron, Deteroit ambassador bridge, Buffalo, Lewiston, Champaign

# 7.In Canada for all 4 months of Jan, Feb, March & April most of the times the highest time taken in the Huntingdon & Andover airports

# 8.In America 95% of the times the most time taken airport is Alexandria bay followed by Alcan & Antler ports.

# 9.In all the 4 months the traffic was more during the week4 of every month and so the more time taken. 