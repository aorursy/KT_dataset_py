import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





sns.set(style="darkgrid")
df = pd.read_csv('../input/montcoalert/911.csv')
df.info()
df.head()
df['station_num'] = df.desc.str.split('Station', expand=True)[1].str.split(';', expand=True)[0]

df.head()
df_bar = df.station_num.str.replace(':', '').value_counts()[:10]

df_station = pd.DataFrame(df_bar)

df_station
#top 5 zip codes

df['zip'].value_counts().head(5)
#top 5 township

df['twp'].value_counts().head(5)
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])
df.head()
df["timeStamp"] = pd.to_datetime(df["timeStamp"])

df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)

df['Month'] = df['timeStamp'].apply(lambda x: x.month)

df['Day of Week'] = df['timeStamp'].apply(lambda x: x.dayofweek)
dmap= {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}

dmonth = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}

df['Month']= df['Month'].map(dmonth)

df['Day of Week']= df['Day of Week'].map(dmap)

df.head()
df_bar = df.station_num.str.replace(':', '').value_counts()[:10]

plt.figure(figsize=(12, 8))

x = list(df_bar.index)

y = list(df_bar.values)

x.reverse()

y.reverse()



plt.title("Most Called Stations")

plt.ylabel("Station")

plt.xlabel("Number of calls")



plt.barh(x, y)
reason = df['Reason'].value_counts()

plt.figure(figsize=(10,5))

sns.barplot(reason.index, reason.values, alpha=0.8 , palette = 'rocket')

plt.title('Main Categories of Reason')

plt.ylabel('Number of calls', fontsize=12)

plt.xlabel('Types of Reason', fontsize=12)

plt.show()
twp = df['twp'].value_counts()[:5]

plt.figure(figsize=(10,5))

sns.barplot(twp.index, twp.values, alpha=0.8)

plt.title('Top 5 cities')

plt.ylabel('Number of calls', fontsize=12)

plt.xlabel('Types of Township', fontsize=12)

plt.show()
plt.figure(figsize=(18,6))

sns.countplot( x='twp',data=df,order=df['twp'].value_counts().index[:10], hue='Reason', palette='rocket')

plt.title('Township wise type of calls')

plt.show()
df["Day/Night"] = df["timeStamp"].apply(lambda x : "Night" if int(x.strftime("%H")) > 18 else "Day")

df.head(4)
df["Day/Night"].value_counts()
plt.figure(figsize=(10,5))

sns.set_context("paper", font_scale =1.5)

sns.countplot(x='Day/Night',data=df,palette='gnuplot')

sns.set_style("darkgrid")
hour  = df['Hour'].value_counts()

plt.figure(figsize=(10,5))

sns.barplot(hour.index, hour.values, alpha=0.8)

plt.title('Number of calls vs hours')

plt.ylabel('Number of Calls', fontsize=12)

plt.xlabel('Hours', fontsize=12)

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(x='Hour',data=df,hue="Reason",palette = 'rocket' )

#plt.legend(loc=[0,1])

plt.title('Hour wise count plot for different types')

plt.show()
month  = df['Month'].value_counts()

plt.figure(figsize=(10,5))

sns.barplot(month.index, month.values, alpha=0.8)

plt.title('Number of calls vs Month')

plt.ylabel('Number of Calls', fontsize=12)

plt.xlabel('Month', fontsize=12)

plt.show()
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

plt.figure(figsize=(12,8))

sns.countplot(x='Day of Week',data=df,hue="Reason",order = days,palette = 'viridis')

plt.legend(loc=[0,1])

plt.title('Day wise count plot for different types')

plt.show()
month = ['January','February','March','April','May','June','July','August','September','October','November','December']

plt.figure(figsize=(12,8))

sns.countplot(x='Month',data=df,hue="Reason",order = month,palette = 'deep') 

#plt.legend(loc=[0,1])

plt.title('Month wise count plot for different types')

plt.show()
plt.xlabel('Traffic Category')

plt.ylabel('Count')

plt.title('Count of Top 5 Emergencies under Traffic Cataegory')

df["title"][df["title"].str.match("Traffic")].value_counts().sort_values(ascending=False).head().plot.bar(color = 'turquoise')

plt.show()
plt.xlabel('Traffic Category')

plt.ylabel('Count')

plt.title('Count of Top 5 Emergencies under Fire Cataegory')

df["title"][df["title"].str.match("EMS")].value_counts().sort_values(ascending=False).head().plot.bar(color = 'blue')

plt.show()
plt.xlabel('Traffic Category')

plt.ylabel('Count')

plt.title('Count of Top 5 Emergencies under Fire Cataegory')

df["title"][df["title"].str.match("Fire")].value_counts().sort_values(ascending=False).head().plot.bar(color = 'red')

plt.show()
#making time series

df['timeStamp'] = pd.to_datetime(df['timeStamp'])

df['Date']=df['timeStamp'].apply(lambda t: t.date())
df.groupby('Date').count()['twp'].plot(linewidth=2 , figsize=(20, 10))



plt.xticks(rotation = 'vertical', size = 10)
df.groupby('Date').count()['twp'][750::].plot(linewidth=2 , figsize=(20, 10) , color = 'g')



plt.xticks(rotation = 'vertical', size = 10)