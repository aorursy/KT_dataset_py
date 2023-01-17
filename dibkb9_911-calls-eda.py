# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#read file

df = pd.read_csv('/kaggle/input/montcoalert/911.csv')
df.head()
print(f'Columns : {df.shape[1]}')

print(f'Rows : {df.shape[0]}')
#dtypes of features

for col in df.columns:

    print(f'{col} : {df[col].dtype}')
#print percentage of missing values

df_miss = pd.DataFrame(columns=['count', 'percetage'])

for col in df.columns:

    df_miss.loc[col] = [df[col].isnull().sum(),(df[col].isnull().sum()/663522)*100]

df_miss    
#visualize missing values

fig,ax = plt.subplots(figsize=(18,6))

ax = sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='inferno')

for label in ax.get_xticklabels():

    label.set_fontsize(12)
#convert 'timeStamp' column to DateTime column

df['timeStamp'] = pd.to_datetime(df['timeStamp'],format='%Y-%m-%d %H:%M:%S', errors='ignore')
print('Zip codes in dataset:')

print(df['zip'].nunique())
df['zip'].value_counts().head(5)
zip_df = pd.DataFrame(df['zip'].value_counts())

zip_df['index']= zip_df.index.astype('object')

zip_df.sort_values(by='zip',ascending=False,inplace=True)

#visualization

sns.set_style('darkgrid')

fig,ax = plt.subplots(figsize=(15,9))

ax = sns.barplot(x ='index',y = 'zip',data=zip_df.head(5))



#labels

ax.set_xlabel('Zip Code', fontsize=18,)

ax.set_ylabel('Total Calls',fontsize=18)

df['twp'].value_counts().head()
twn_df = pd.DataFrame(df['twp'].value_counts())

twn_df['town'] = twn_df.index

my_range=list(range(len(twn_df.head(9).index)+1))

#visualization

sns.set_style('darkgrid')

fig,ax = plt.subplots(figsize=(15,9))

ax = sns.countplot(y='twp',data=df,order=df['twp'].value_counts().head(12).index,palette='rocket')



#labels

ax.set_ylabel('', fontsize=15,)

ax.set_xlabel('Total Calls',fontsize=15)



#annotations    

for p in ax.patches:

    ax.annotate(int(p.get_width()),((p.get_x() + p.get_width()), p.get_y()),

                xytext=(9,-15),fontsize=9,color='#000000',textcoords='offset points', ha='left',va='center',size=12)
#create a new dataframe 'reason'

df['reason'] = df['title'].apply(lambda title: title.split(':')[0])



#create a new dataframe 'title_category'

df['title_category'] = df['title'].apply(lambda title: title.split(':')[1])
fig,ax = plt.subplots(figsize=(15,9))

ax = sns.countplot(y='twp',data=df,order=df['twp'].value_counts().head(9).index,hue='reason')
print("Unique emergency codes: ")

print(df['title'].nunique())
reason_df = df.groupby('reason').count()

reason_df['label'] = reason_df.index

reason_df = reason_df.sort_values(by='title_category',ascending = False)
sns.set_style('white')

fig,ax = plt.subplots(ncols=2,figsize=(18,6))



xrange = np.arange(1,len(reason_df.sum())+1)



#donut chart

theme = plt.get_cmap('Blues_r')

ax[0].set_prop_cycle("color", [theme(1. * i / len(reason_df))for i in range(len(reason_df))])    

wedges, texts,_ = ax[0].pie(reason_df['title_category'], wedgeprops=dict(width=0.45), startangle=-90,labels=reason_df.index,

                  autopct="%.1f%%",textprops={'fontsize': 18,

                                             'color':'#000000'})



 





ax[1] = sns.barplot(x = 'title_category',y ='label',data =reason_df,palette = 'Blues_r')

my_range=list(range(len(reason_df.index)+1))



#labels

ax[1].set_ylabel('', fontsize=15)

ax[1].set_xlabel('Total Calls',fontsize=15)



for tick in ax[1].get_yticklabels():

    tick.set_fontsize(15)

for tick in ax[1].get_xticklabels():

    tick.set_fontsize(12)

sns.despine(right=True, left=True, bottom=True)    



#annotations    

for x,y in zip(my_range,reason_df["title_category"]):

    label = "{:}".format(y)

    plt.annotate(label, # this is the text

                 (y,x), # this is the point to label

                  textcoords="offset points",# how to position the text

                 xytext=(9,0), # distance from text to points (x,y)

                 ha='left',va="center",size=15) 

plt.tight_layout()   
fig,ax = plt.subplots(figsize=(15,9))

ax = sns.countplot(y='title',data=df,order=df['title'].value_counts().head(12).index,palette='inferno')

#labels

ax.set_ylabel('')

ax.set_xlabel('Total Calls',fontsize=18)

#despine

sns.despine(right=True, left=True, bottom=True)   

#annotations    

for p in ax.patches:

    ax.annotate(int(p.get_width()),((p.get_x() + p.get_width()), p.get_y()),

                xytext=(9,-15),fontsize=9,color='#000000',textcoords='offset points', ha='left',va='center',size=12)
def plot_category(cat):

    filt_ems = df['reason'] == cat

    fig,ax = plt.subplots(figsize=(15,9))

    ax = sns.countplot(y='title_category',data=df.loc[filt_ems],

                       order=df.loc[filt_ems]['title_category'].value_counts().head(12).index,palette='Reds_r')

    #labels

    ax.set_ylabel('')

    ax.set_xlabel('Total Calls',fontsize=18)

    ax.set_title(f'{cat}',fontsize=18)

    #despine

    sns.despine(right=True, left=True, bottom=True)   

    #annotations    

    for p in ax.patches:

        ax.annotate(int(p.get_width()),((p.get_x() + p.get_width()), p.get_y()),

                    xytext=(9,-15),fontsize=9,color='#000000',textcoords='offset points', ha='left',va='center',size=12)
plot_category('EMS')
plot_category('Fire')
plot_category('Traffic')
import calendar 

import datetime as dt
#create 'month','dayofweek' and 'hour' columns

df['month'] = df['timeStamp'].apply(lambda x: x.month)

df['dayofweek'] = df['timeStamp'].apply(lambda x: x.weekday())

df['hour'] = df['timeStamp'].apply(lambda x: x.hour)

df['date'] = df['timeStamp'].apply(lambda x: x.date())

df['year'] = df['timeStamp'].apply(lambda x: x.year)

#create another column 'Day'

df['day'] = df['dayofweek'].apply(lambda x: calendar.day_name[x])

#create another column 'Month'

df['month_name'] = df['month'].apply(lambda x: calendar.month_name[x])
fig,ax = plt.subplots(figsize=(18,9))

ax = sns.countplot(x='month_name',data=df.sort_values(by='month'),palette='inferno')

#despine

sns.despine(right=True, left=True, bottom=True)   

#annotations    

for p in ax.patches:

    ax.annotate(format(p.get_height(), '.2f'), 

                   (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center', va = 'center', 

                   xytext = (0, 9), 

                   textcoords = 'offset points',size=12)

for tick in ax.get_yticklabels():

    tick.set_fontsize(12)

for tick in ax.get_xticklabels():

    tick.set_fontsize(15) 

#labels

ax.set_xlabel('')

ax.set_ylabel('Total Calls',fontsize=18)
#groupby 'month' and 'reason' 

reason_month_grp = df.groupby(['reason','month']).count()

reason_month_grp =reason_month_grp.reset_index()
#split each reason 

ems_month = reason_month_grp[reason_month_grp['reason'] == 'EMS']

fire_month = reason_month_grp[reason_month_grp['reason'] == 'Fire']

traffic_month = reason_month_grp[reason_month_grp['reason'] == 'Traffic']
fig,ax = plt.subplots(figsize=(18,9))

plt.style.use('fivethirtyeight')

ax = sns.lineplot(x='month',y='lat',data = ems_month,label='EMS')

ax = sns.lineplot(x='month',y='lat',data = fire_month,label='Fire')

ax = sns.lineplot(x='month',y='lat',data = traffic_month,label='Traffic')

sns.despine(right=True, left=True, bottom=True) 

ax.set_ylabel('')

ax.set_title('Time-Series plot Monthy calls')
sns.set_style('white')



fig,ax = plt.subplots(figsize=(18,9))

ax = sns.countplot(x='day',data=df.sort_values(by='dayofweek'),palette='viridis')

#despine

sns.despine(right=True, left=True, bottom=True)   

#annotations    

for p in ax.patches:

    ax.annotate(format(p.get_height(), '.2f'), 

                   (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center', va = 'center', 

                   xytext = (0, 9), 

                   textcoords = 'offset points',size=15)

for tick in ax.get_yticklabels():

    tick.set_fontsize(12)

for tick in ax.get_xticklabels():

    tick.set_fontsize(15) 

#labels

ax.set_xlabel('')

ax.set_ylabel('Total Calls',fontsize=18)
#groupby 'dayofweek' and 'reason' 

reason_day_grp = df.groupby(['reason','dayofweek']).count()

reason_day_grp =reason_day_grp.reset_index()
#split each reason 

ems_day = reason_day_grp[reason_day_grp['reason'] == 'EMS']

fire_day = reason_day_grp[reason_day_grp['reason'] == 'Fire']

traffic_day = reason_day_grp[reason_day_grp['reason'] == 'Traffic']
sns.set_style('white')

fig,ax = plt.subplots(figsize=(18,9))

plt.style.use('fivethirtyeight')

ax = sns.lineplot(x='dayofweek',y='lat',data = ems_day,label='EMS')

ax = sns.lineplot(x='dayofweek',y='lat',data = fire_day,label='Fire')

ax = sns.lineplot(x='dayofweek',y='lat',data = traffic_day,label='Traffic')

sns.despine(right=True, left=True, bottom=True) 

ax.set_ylabel('')

ax.set_title('Time-Series plot Daily calls')
sns.set_style('white')



fig,ax = plt.subplots(figsize=(18,9))

ax = sns.countplot(x='hour',data=df.sort_values(by='hour'),palette='twilight_r')

#despine

sns.despine(right=True, left=True, bottom=True)   

#annotations    

for p in ax.patches:

    ax.annotate(format(p.get_height(), '.2f'), 

                   (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center', va = 'center', 

                   xytext = (0, 9), 

                   textcoords = 'offset points',size=12)

for tick in ax.get_yticklabels():

    tick.set_fontsize(12)

for tick in ax.get_xticklabels():

    tick.set_fontsize(15) 

#labels

ax.set_xlabel('Hour',fontsize=18)

ax.set_ylabel('Total Calls')

#groupby 'hour' and 'reason' 

reason_hour_grp = df.groupby(['reason','hour']).count()

reason_hour_grp =reason_hour_grp.reset_index()
#split each reason 

ems_hour = reason_hour_grp[reason_hour_grp['reason'] == 'EMS']

fire_hour = reason_hour_grp[reason_hour_grp['reason'] == 'Fire']

traffic_hour = reason_hour_grp[reason_hour_grp['reason'] == 'Traffic']
sns.set_style('white')

fig,ax = plt.subplots(figsize=(18,9))

plt.style.use('fivethirtyeight')

ax = sns.lineplot(x='hour',y='lat',data = ems_hour,label='EMS')

ax = sns.lineplot(x='hour',y='lat',data = fire_hour,label='Fire')

ax = sns.lineplot(x='hour',y='lat',data = traffic_hour,label='Traffic')

sns.despine(right=True, left=True, bottom=True) 

ax.set_ylabel('')

ax.set_title('Time-Series plot Hourly calls')
#create date dataframe

df_date = df.groupby(['date','reason']).count()

df_date = df_date.reset_index()
df_date_ems = df_date[df_date['reason']=='EMS']

df_date_fire = df_date[df_date['reason']=='Fire']

df_date_traffic = df_date[df_date['reason']=='Traffic']
sns.set_style('darkgrid')

fig,ax = plt.subplots(figsize=(18,9))

plt.style.use('seaborn')

ax = sns.lineplot(x='date',y='lat',data = df_date_ems,label='EMS')

sns.despine(right=True, left=True, bottom=True) 

ax.set_ylabel('')

ax.set_xlabel('')

ax.set_title('Time-Series plot of EMS calls',size=15)

for tick in ax.get_yticklabels():

    tick.set_fontsize(12)

for tick in ax.get_xticklabels():

    tick.set_fontsize(15) 
sns.set_style('darkgrid')

fig,ax = plt.subplots(figsize=(18,9))

plt.style.use('seaborn')

ax = sns.lineplot(x='date',y='lat',data = df_date_fire,label='Fire',color="#e37d00")

sns.despine(right=True, left=True, bottom=True) 

ax.set_ylabel('')

ax.set_xlabel('')

ax.set_title('Time-Series plot of Fire calls',size=15)

for tick in ax.get_yticklabels():

    tick.set_fontsize(12)

for tick in ax.get_xticklabels():

    tick.set_fontsize(15) 
sns.set_style('darkgrid')

fig,ax = plt.subplots(figsize=(18,9))

plt.style.use('seaborn')

ax = sns.lineplot(x='date',y='lat',data = df_date_traffic,label='Traffic',color="#b30000")

sns.despine(right=True, left=True, bottom=True) 

ax.set_ylabel('')

ax.set_xlabel('')

ax.set_title('Time-Series plot of Traffic calls',size=15)

for tick in ax.get_yticklabels():

    tick.set_fontsize(12)

for tick in ax.get_xticklabels():

    tick.set_fontsize(15) 
#create 'day' X 'hour' matrix

df_day_hour = df.groupby(['dayofweek','hour']).count()['lat'].unstack()
dict_weekday={

    0:'Monday',

    1:'Tuesday',

    2:'Wednesday',

    3:'Thursday',

    4:'Friday',

    5:'Saturday',

    6:'Sunday'

}
#rename index

df_day_hour = df_day_hour.rename(index=dict_weekday)
#plot heatmap

fig,ax = plt.subplots(figsize=(18,9))

ax = sns.heatmap(df_day_hour,annot=False,cmap="OrRd",)

for item in ax.get_yticklabels():

    item.set_rotation(0)

ax.set_xlabel('')

ax.set_ylabel('')



#ticksize

for tick in ax.get_xticklabels():

    tick.set_fontsize(15)

for tick in ax.get_yticklabels():

    tick.set_fontsize(15)  
#create 'day' X 'hour' matrix

df_month_day = df.groupby(['dayofweek','month']).count()['lat'].unstack()
dict_month = {

    1:'January',

    2:'February',

    3:'March',

    4:'April',

    5:'May',

    6:'June',

    7:'July',

    8:'August',

    9:'September',

    10:'October',

    11:'November',

    12:'December',

}
#rename index

df_month_day = df_month_day.rename(index=dict_weekday)

df_month_day = df_month_day.rename(columns=dict_month)
#plot heatmap

fig,ax = plt.subplots(figsize=(18,9))

ax = sns.heatmap(df_month_day.transpose(),annot=False,cmap="OrRd",)

for item in ax.get_yticklabels():

    item.set_rotation(0)

ax.set_xlabel('')

ax.set_ylabel('')

#ticksize

for tick in ax.get_xticklabels():

    tick.set_fontsize(15)

for tick in ax.get_yticklabels():

    tick.set_fontsize(15) 
#create 'year' X 'month' matrix

df_year_month = df.groupby(['year','month']).count()['lat'].unstack()

#rename

df_year_month.rename(columns=dict_month,inplace=True)
#plot heatmap

fig,ax = plt.subplots(figsize=(18,9))

ax = sns.heatmap(df_year_month,annot=False,cmap="OrRd",)

for item in ax.get_yticklabels():

    item.set_rotation(0)

ax.set_xlabel('')

ax.set_ylabel('')

#ticksize

for tick in ax.get_xticklabels():

    tick.set_fontsize(15)

for tick in ax.get_yticklabels():

    tick.set_fontsize(15)    