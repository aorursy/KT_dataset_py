# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium.plugins import HeatMap
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import numpy as np, matplotlib.pyplot as plt, seaborn as sns



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
crimes=pd.read_csv("../input/crimes-in-boston/crime.csv",encoding = "ISO-8859-1")
crimes.head()
crimes.columns = map(str.lower, crimes.columns)
crimes.head()
print(crimes.shape, crimes.drop_duplicates().shape)
crimes = crimes.drop_duplicates() #It has 23 duplicates.
import missingno as msno
msno.matrix(crimes)
print(crimes.isnull().sum(), end = '\n\n')
print(crimes[(crimes['lat'].isnull()) | (crimes['long'].isnull())]['location'].unique())
off = crimes.groupby('offense_code_group')['offense_code'].nunique().sort_values(ascending = False)
off.to_frame().reset_index()[:10]
y=crimes.year.value_counts()
print(y)
sns.set(style="whitegrid", color_codes=True)

year_counts = crimes.groupby('year').count()['incident_number'].to_frame().reset_index()
pal = sns.color_palette("GnBu_d", len(year_counts['year']))
rank = year_counts["incident_number"].argsort().argsort() 
sns.set(rc={'figure.figsize':(5,5)})
ax = sns.barplot(x = 'year' , y="incident_number", data = year_counts, palette=np.array(pal[::-1])[rank])

plt.ylim(40000,110000)
plt.show()
month_counts = crimes.groupby('month').count()['incident_number'].to_frame().reset_index()
sns.set(rc={'figure.figsize':(8,5)})
pal = sns.color_palette("Blues_d", len(month_counts['month']))
rank = month_counts["incident_number"].argsort().argsort() 
ax = sns.barplot(x = 'month' , y="incident_number", data = month_counts,palette=np.array(pal[::-1])[rank])
plt.figure(figsize=(15,7))
crimes.groupby(['year','month']).count()['incident_number'].plot.bar()
day_counts = crimes.groupby('day_of_week').count()['incident_number'].to_frame().reset_index()
sns.set(rc={'figure.figsize':(8,8)})
pal = sns.color_palette("ch:3.1,-.2,dark=.2", len(day_counts['day_of_week']))
rank = day_counts["incident_number"].argsort().argsort() 
ax = sns.barplot(x = 'day_of_week' , y="incident_number", data = day_counts, palette=np.array(pal[::])[rank])
hour_nums = crimes.groupby(['hour']).count()['incident_number'].to_frame().reset_index()
sns.set(rc={'figure.figsize':(20,6)})
pal = sns.color_palette("ch:3.2,-.2,dark=.2", len(hour_nums['hour']))
rank = hour_nums["incident_number"].argsort().argsort() 
ax = sns.barplot(x = 'hour' , y="incident_number", data = hour_nums, palette=np.array(pal[::])[rank])

# ax.set_xticklabels(ax.get_xticklabels(),rotation=90); #x eks.deki değerleri yatay yazar.
district_nums =  crimes.groupby('district').count()['incident_number'].sort_values(ascending = False).to_frame().reset_index()
sns.set(rc={'figure.figsize':(7,5)})
pal = sns.color_palette("ch:3.4,-.2,dark=.2", len(district_nums['district']))
rank = district_nums["incident_number"].argsort().argsort() 
ax = sns.barplot(x = 'district' , y="incident_number", data = district_nums, palette=np.array(pal[::])[rank])
print(crimes.groupby('year').count()['incident_number'])
sns.set(rc={'figure.figsize':(5,5)})
pal = sns.color_palette("ch:3.5,-.2,dark=.2", len(year_counts['year']))
rank = year_counts["incident_number"].argsort().argsort() 
ax = sns.barplot(x = 'year' , y="incident_number", data = year_counts, palette=np.array(pal[::])[rank])

print('Count of Months Per Year:\n',crimes.groupby('year')['month'].nunique())
av_month = (crimes.groupby('year').count()['incident_number'] / crimes.groupby('year')['month'].nunique()).to_frame().reset_index()
print('\nAverage monthly incident per year:\n',av_month)
av_month.rename(columns = {0:'incident_number'}, inplace = True)

pal = sns.color_palette("ch:3.7,-.3,dark=.3", len(year_counts['year']))
rank = av_month["incident_number"].argsort().argsort() 
sns.set(rc={'figure.figsize':(5,5)})
ax = sns.barplot(x = 'year' , y="incident_number", data = av_month, palette=np.array(pal[::])[rank])
print(crimes.min()['occurred_on_date'])
print(crimes.max()['occurred_on_date'])
crimes['occurred_on_date'] = pd.to_datetime(crimes['occurred_on_date'])
yearly_counts = crimes.groupby('year').count()['incident_number'].to_numpy()
days = []
for year in crimes.year.sort_values().unique():
    days.append((crimes[crimes['year'] == year].max().occurred_on_date - crimes[crimes['year'] == year].min().occurred_on_date).days)

average_daily_incidents = yearly_counts / days  
    
print([str(year)+ ": "+ str(avg)[:4] for year, avg in enumerate(average_daily_incidents, 2015)])
d_avg = pd.DataFrame(data = average_daily_incidents, index = av_month.index)
d_avg.rename(columns = {0:'counts'}, inplace = True)

pal = sns.color_palette("ch:3.8,-.1,dark=.2", len(year_counts['year']))
rank = av_month["incident_number"].argsort().argsort() 
sns.set(rc={'figure.figsize':(5,5)})
ax = sns.barplot(x = d_avg.index , y="counts", data = d_avg, palette=np.array(pal[::])[rank])
ax.set_xticklabels(crimes.year.sort_values().unique());
shooting_years
shooting = crimes.dropna(subset = ['shooting'])
shooting_years = shooting.groupby('year').count()['incident_number'].to_frame().reset_index()

sns.set(rc={'figure.figsize':(5,5)})
pal = sns.color_palette("ch:4.2,-.1,dark=.2, light=.8", len(shooting_years['year']))

sns.set(rc={'figure.figsize':(5,5)})
ax = sns.barplot(x = shooting_years["year"] , y="incident_number", data = shooting_years, palette=np.array(pal[::])[rank])
# Now let's look at the top 10 places to see where the crime was committed the most.
""" stmax = df.STREET.value_counts().head(10)
print(stmax)
print(stmax["CENTRE ST"])"""
# Now let's visualize it.
''' sns.catplot(y='STREET',
           kind='count',
           height=7, 
           aspect=2,
           order=df.STREET.value_counts().head(10).index,
           data=df) '''
# In the graph below, we see the 30 most common crimes in the city. Top 3 crimes are: Motor Vehicle,  Accident Response, Larceny, Medical Assistance.
''' df2 = pd.DataFrame(columns = ['offenses'])
df2["offenses"]=[each for each in crimes.offense_code_group.unique()]
df2["count"]=[len(crimes[crimes.offense_code_group==each]) for each in df2.offenses]
df2=df2.sort_values(by=['count'],ascending=False)

plt.figure(figsize=(25,15))
sns.barplot(x=df2.offenses.head(30), y=df2.value_counts().head(30)
plt.xticks(rotation= 90)
plt.xlabel('offenses')
plt.ylabel('count')
plt.show() '''