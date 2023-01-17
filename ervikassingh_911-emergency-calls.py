import pandas as pd

import datetime as dt

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

font = {

    'size': 'large',

    'weight': 'bold'

}
data=pd.read_csv('../input/911.csv')
data.head()
def c_type(x):

    x=x.split(':')

    return x[0]
data['call_type']=data['title'].apply(c_type)
data.head()
type_data=data.groupby('call_type')['call_type'].count()
sns.set(rc={'figure.figsize':(5,5)})

type_data.plot(kind='bar')

plt.xticks(rotation=0)

plt.savefig('type.png')
data.info()
data['timeStamp'] = pd.to_datetime(data['timeStamp'])
data.head()
data.info()
data['year'] = data['timeStamp'].dt.year

data['month'] = data['timeStamp'].dt.month_name()

data['day'] = data['timeStamp'].dt.day_name()

data['hour'] = data['timeStamp'].dt.hour
d1=data.groupby(['month','call_type'])['call_type'].count()
d1.head()
d11 = d1.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))
d11.head()
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

d11=d11.reindex(month_order, level=0)
sns.set(rc={'figure.figsize':(15,10)})

d11.unstack().plot(kind='bar')

plt.xticks(rotation=90)

plt.xlabel('Month',fontdict=font)

plt.ylabel('Call Type Percent',fontdict=font)

plt.title('MONTHWISE CALL TYPE',fontdict=font)

plt.savefig('monthly.png')
d2=data.groupby(['day','call_type'])['call_type'].count()
d2.head()
d22 = d2.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))
d22.head()
day_order = ['Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

d22=d22.reindex(day_order, level=0)
d22.unstack().plot(kind='bar')

plt.xticks(rotation=90)

plt.xlabel('Day',fontdict=font)

plt.ylabel('Call Type Percent',fontdict=font)

plt.title('DAYWISE CALL TYPE',fontdict=font)

plt.savefig('daily.png')
def e_type(x):

    x=x.split(':')

    return x[1]
data['emergency_type']=data['title'].apply(e_type)
data.head()
d3=data.groupby(['call_type','emergency_type'])['emergency_type'].count()
d3.head()
d33 = d3.groupby(level=0).apply(lambda x: round(100*x/float(x.sum()),2))
d33.head()
sns.set(rc={'figure.figsize':(10,10)})

plt.pie(d33['EMS'][:5],labels=d33['EMS'][:5].index,autopct='%.2f')

plt.title('First 5 EMS Percentage',fontdict=font)

plt.savefig('EMS.png')
plt.pie(d33['Fire'][:5],labels=d33['Fire'][:5].index,autopct='%.2f')

plt.title('First 5 Fire Percentage',fontdict=font)

plt.savefig('Fire.png')
plt.pie(d33['Traffic'][:5],labels=d33['Traffic'][:5].index,autopct='%.2f')

plt.title('First 5 Traffic Percentage',fontdict=font)

plt.savefig('Traffic.png')