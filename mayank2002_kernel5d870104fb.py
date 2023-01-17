import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import datetime as dt
data = pd.read_csv("../input/911.csv")
data.head()
data =data.dropna()
data['title'].nunique()
def call_type(x):

    if (('EMS' in x)==True):

        return 'EMS'

    elif (('Fire' in x)==True):

        return 'Fire'

    elif (('Traffic' in x)==True):

        return 'Traffic'
data['call_type'] = data['title'].apply(call_type)
data.head(10)
data['call_type'].unique()
# sns.set()


data['call_type'].value_counts().plot.pie(autopct='%.2f%%')

plt.savefig('Call_types.png')
data['call_type'].value_counts()
data['timeStamp'] = pd.to_datetime(data['timeStamp'])
def date_extract(x):

    year = x.year

    month = x.month

    day = x.day

    new_date = str(year)+'-'+str(month)+'-'+str(day)

    return new_date
data['date'] = data['timeStamp'].apply(date_extract)
data.head()
def time_extract(x):

    hour = x.hour

    minute = x.minute

    sec = x.second

    new_time = str(hour)+':'+str(minute)+':'+str(sec)

    return new_time

    
data['time'] = data['timeStamp'].apply(time_extract)
data.head()
data['title'].value_counts()
import calendar

def year_extract(x):

    year = x.year

    return year

def month_extract(x):

    month_num = x.month

    monthname = calendar.month_name[month_num]

    return monthname
data['year'] = data['timeStamp'].apply(year_extract)

data['month'] = data['timeStamp'].apply(month_extract)
data.head()
plot_data = data.groupby('year')['call_type'].value_counts().plot(kind='bar')

for p in plot_data.patches:

    plot_data.annotate(str(p.get_height()), xy = (p.get_x(), p.get_height()))



plt.title('Call/Year',fontsize=35,weight='bold')

plt.xticks(rotation=20)

plt.xlabel('Year,Call_type',fontsize=18,weight='bold')

plt.ylabel('No. of Calls',fontsize=18,weight='bold')

plt.savefig('Call_per_Year.png')
month_data = data.groupby('month')['call_type'].value_counts()

month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']

month_data = month_data.reindex(month_order,level=0)

month_data = month_data.reindex(['EMS','Traffic','Fire'],level=1)

month_data.head()
fig = plt.figure(figsize=(20,4))

month_data.unstack().plot(kind='bar')

plt.title('Calls/Month',fontsize=30,weight='bold')

plt.xlabel('Month',fontsize = 20,weight='bold')

plt.ylabel('No. of Calls',fontsize = 20,weight='bold')

plt.savefig('Calls_per_month.png',bbox_inches='tight')
def get_hour(x):

    hour = x.hour

    return hour
data['hour'] = data['timeStamp'].apply(get_hour)
data.head()
hourly_distribution = data.groupby(['hour','call_type'])['call_type'].count()
hourly_distribution.head()
hourly_distribution = hourly_distribution.reindex(['EMS','Traffic','Fire'],level=1)
hourly_distribution.head()
hourly_distribution.index
x=hourly_distribution.unstack().head()
x.index
font={

    'size':'30',

    'weight':'bold'

}
# plt.figure(figsize=(40,6))

sns.set(rc={'figure.figsize':(18, 8)})

hourly_distribution.unstack().plot.bar()

plt.ylabel('No. of Calls',fontdict=font)

plt.xlabel('Hours',fontdict=font)

plt.title('Calls/hour',fontdict=font)

plt.savefig('Calls_per_hours.png')
hour_percentage = hourly_distribution.groupby(level=0).apply(lambda x : x*100/(x.sum()))

hour_percentage.head()
hour_percentage.unstack().plot(kind='bar')

plt.title('Call/Hour',fontsize = 35,weight='bold')

plt.ylabel('Percentage of Calls',fontsize =18,weight='bold')

plt.xlabel('Hour',fontsize =18,weight='bold')

plt.savefig('percentage_calperhour.png')
def emergency_types(x):

    x = x.split(':')

    x=x[1]

    return(x)
data['Emergency_types'] = data['title'].apply(emergency_types)
data.head(2)
data['Emergency_types'].nunique()
emergency_data = data.groupby('Emergency_types')['call_type'].value_counts()
emergency_top = emergency_data.sort_values(ascending=False)
emergency_top.head(15).unstack().plot(kind='bar')

plt.title('Calls/Emergency Types',fontdict=font)

plt.xlabel('Emergency Type',fontsize = 20,weight='bold')

plt.ylabel('Calls',fontsize = 20,weight='bold')

plt.savefig('Calls_per_EmergencyTypes.png',bbox_inches='tight')
ems_emer_data = data.groupby(['Emergency_types'])['call_type'].value_counts()

ems_emer_data = ems_emer_data.unstack()

ems_emer_top = ems_emer_data['EMS'].sort_values(ascending =False)

ems_emer_top = ems_emer_top.head(20)
ems_emer_top.plot(kind='bar')

plt.title('EMS Calls / Emergency_types',fontdict=font)

plt.xlabel('Emergency Types ',fontsize = 18,weight = 'bold')

plt.ylabel('No. of EMS calls ',fontsize = 18,weight = 'bold')

plt.savefig('EMSCalls_per_EmergencyTypes.png',bbox_inches='tight')
traffic_emer_data = data.groupby(['Emergency_types'])['call_type'].value_counts()

traffic_emer_data = traffic_emer_data.unstack()

traffic_emer_top = traffic_emer_data['Traffic'].sort_values(ascending =False)

traffic_emer_top = traffic_emer_top.head(10)
traffic_emer_top.plot(kind='bar')

plt.title('Traffic Calls / Emergency_types',fontdict=font)

plt.xlabel('Emergency Types ',fontsize = 18,weight = 'bold')

plt.ylabel('No. of Traffic calls ',fontsize = 18,weight = 'bold')

plt.savefig('Traffic_Calls_per_EmergencyTypes.png',bbox_inches='tight')
fire_emer_data = data.groupby(['Emergency_types'])['call_type'].value_counts()

fire_emer_data = fire_emer_data.unstack()

fire_emer_top = fire_emer_data['Fire'].sort_values(ascending =False)

fire_emer_top = fire_emer_top.head(15)
fire_emer_top.plot(kind='bar')

plt.title('Fire Calls / Emergency_types',fontdict=font)

plt.xlabel('Emergency Types ',fontsize = 18,weight = 'bold')

plt.ylabel('No. of Fire calls ',fontsize = 18,weight = 'bold')

plt.savefig('FIre_Calls_per_EmergencyTypes.png',bbox_inches='tight')
data['day'] = data['timeStamp'].dt.day_name()
data.head(2)
day_data = data.groupby('day')['call_type'].value_counts()
day_data = day_data.reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],level=0)
day_data.unstack().plot(kind='bar')

plt.title('Calls/Day',fontdict=font)

plt.xlabel('Days',fontsize = 20,weight='bold')

plt.ylabel('No. of Calls',fontsize = 20,weight='bold')

plt.savefig('Calls_per_day.png',bbox_inches='tight')