#load necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import matplotlib as mpl

import matplotlib.patches as mpatches



#load data files



    

df7_1=pd.read_csv("../input/toronto-bikeshare-data/bikeshare-ridership-2017/2017 Data/Bikeshare Ridership (2017 Q1).csv",encoding="ISO 8859-1")

df7_2=pd.read_csv("../input/toronto-bikeshare-data/bikeshare-ridership-2017/2017 Data/Bikeshare Ridership (2017 Q2).csv",encoding="ISO 8859-1")

df7_3=pd.read_csv("../input/toronto-bikeshare-data/bikeshare-ridership-2017/2017 Data/Bikeshare Ridership (2017 Q3).csv",encoding="ISO 8859-1")

df7_4=pd.read_csv("../input/toronto-bikeshare-data/bikeshare-ridership-2017/2017 Data/Bikeshare Ridership (2017 Q4).csv",encoding="ISO 8859-1")



df8_1=pd.read_csv("../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q1 2018.csv",encoding="ISO 8859-1")

df8_2=pd.read_csv("../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q2 2018.csv",encoding="ISO 8859-1")

df8_3=pd.read_csv("../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q3 2018.csv",encoding="ISO 8859-1")

df8_4=pd.read_csv("../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q4 2018.csv",encoding="ISO 8859-1")



#add datafiles together to create one big file

df_list=[df7_1,df7_2,df7_3,df7_4,df8_1,df8_2,df8_3,df8_4]



df=pd.concat(df_list,sort=False)

#reset index and remove the old index column

df.reset_index(inplace=True)

df.drop('index',axis=1, inplace=True)
#to disable python warnings

pd.options.mode.chained_assignment = None 

#split trip_start_time column into start_time and start_date

df['start_date'],df['start_time']=[x.split(' ')[0] for x in df['trip_start_time']],[x.split(' ')[1] for x in df['trip_start_time']]



#trip_stop_time requires some more transformation

#looks like some of the rows do not let to split them; let's find those rows and delete them

no_date=[x for x in df['trip_stop_time'] if not ' ' in x]



df.drop(df[df['trip_stop_time']==no_date[0]].index, axis=0, inplace=True)

df.reset_index(inplace=True)

df['stop_date'],df['stop_time']=[x.split(' ')[0] for x in df['trip_stop_time']],[x.split(' ')[1] for x in df['trip_stop_time']]



#since there are some dates which are formated by european style and all the others are formated by the USA style

#it is easy to find the dates where month is bigger than 12, however how to figure out which dates where the first and the second number is lower than 12

#let's assume that these numbers belong to one dataset part; max index value is 453082; rows in the first df7_1 of the dataset is 132123 and the second part of the dataset df7_2 is 333353 rows long

#in total it is 465476 rows; it is slightly higher than the max value. so, let's transform the first two datasets



#find the final index value for transformation

margin=df7_1.shape[0]+df7_2.shape[0]

#create a list of properly transformed dates

Date = ['{}/{}/{}'.format(m,d,y) for d, m, y in map(lambda x: x.split('/'), df['start_date'])]



#substitute old mixed values with the new saved to Data list

df['start_date'][:margin]=Date[:margin]

#some of the year values are two digit and not 4 digit as the rest



date_to_fix=[]

for i,val in enumerate(df['start_date'].unique()):

    if len(val.split('/')[2])<4:

        date_to_fix.append(val)

#find index of the rows with short year

bad_date_index=df[df['start_date'].isin(date_to_fix)].index

#fix the problem

df['start_date'][bad_date_index]=[x[:6]+'20'+x[-2:] for x in df['start_date'][bad_date_index]]



#create list of unique date not transformed to datetime for further need in the future

start=df['start_date'].unique()

#transform the column to datetime

df['start_date']=[datetime.datetime.strptime(x,'%m/%d/%Y') for x in df['start_date']]



#the same for the stop_date column

margin=df7_1.shape[0]+df7_2.shape[0]

Date = ['{}/{}/{}'.format(m,d,y) for d, m, y in map(lambda x: x.split('/'), df['stop_date'])]

df['stop_date'][:margin]=Date[:margin]



date_to_fix=[]

for i,val in enumerate(df['stop_date'].unique()):

    if len(val.split('/')[2])<4:

        date_to_fix.append(val)

len(date_to_fix)



bad_date_index=df[df['stop_date'].isin(date_to_fix)].index

df['stop_date'][bad_date_index]=[x[:6]+'20'+x[-2:] for x in df['stop_date'][bad_date_index]]



stop=df['stop_date'].unique()

#still stop_date does not let to convert to datetime; let's look for mistakes

#this is going to help me to see which values do not let to convert it to datetime format

start_not=[x for x in start if x not in stop]

stop_not=[x for x in stop if x not in start]



#it turned out that stop_date has incorrect date; let's delete it

df.drop(df[df['stop_date']=='1/01/12018'].index, axis=0, inplace=True)

df.reset_index(inplace=True)

df.drop('index',axis=1, inplace=True)



df['stop_date']=[datetime.datetime.strptime(x,'%m/%d/%Y') for x in df['stop_date']]



#now we need to make time columns of datetime format



df['start_day_week']=[x.weekday() for x in df['start_date']]

df['stop_day_week']=[x.weekday() for x in df['stop_date']]



#create weekday vocabulary

day_voc={0:'Sunday',1:'Monday',2:'Tuesday', 3:'Wednesday', 4:'Thursday', 5:'Friday',6:'Saturday' }



#create columns with weekdays full names

df['start_day_week']=df['start_day_week'].map(day_voc)

df['stop_day_week']=df['stop_day_week'].map(day_voc)



#some of the time rows consist of minutes and hours and some of them consist of minutes, hours and seconds

long_time_start=[]

for i,val in enumerate(df['start_time'].unique()):

    if len(val.split(':'))>2:

        long_time_start.append(val)



long_time_stop=[]

for i,val in enumerate(df['stop_time'].unique()):

    if len(val.split(':'))>2:

        long_time_stop.append(val)





long_time_start_index=df[df['start_time'].isin(long_time_start)].index

long_time_stop_index=df[df['stop_time'].isin(long_time_stop)].index



df['start_time'][long_time_start_index]=['{}:{}'.format(h, m) for h,m,s in map(lambda x: x.split(':') ,df['start_time'][long_time_start_index])]

df['stop_time'][long_time_stop_index]=['{}:{}'.format(h, m) for h,m,s in map(lambda x: x.split(':') ,df['stop_time'][long_time_stop_index])]



df['start_time']=[datetime.datetime.strptime(x,'%H:%M').time() for x in df['start_time']]

df['stop_time']=[datetime.datetime.strptime(x,'%H:%M').time() for x in df['stop_time']]
#let's create columns month and year for better visualisation and filtering

df['month_start']=[x.month for x in df['start_date']]

df['month_stop']=[x.month for x in df['stop_date']]



month_dict={1:'January', 2:'February', 3:'March', 4:'April',5:'May', 6:'June', 7:'July', 8:'August',9:'September', 10:'October', 11:'November', 12:'December'}

df['month_start']=df['month_start'].map(month_dict)

df['month_stop']=df['month_stop'].map(month_dict)



#year columns

df['month_start'].unique()

index_2017=[x.year==2017 for x in df['start_date']]

df['year']=df['month_start']

df['year'][index_2017]= '2017'

index_2018=[x.year==2018 for x in df['start_date']]

df['year'][index_2018]= '2018'



#divide dataset to 2017 and 2018 to use later for visualisations

df['month_start'].unique()

index_2017=[x.year==2017 for x in df['start_date']]

df_2017 = df[index_2017]

index_2018=[x.year==2018 for x in df['start_date']]

df_2018=df[index_2018]

fig, ax = plt.subplots(1, 2, figsize=(18,4))

sns.set_style( {'axes.spines.left': False,

 'axes.spines.bottom': False,

 'axes.spines.right': False,

 'axes.spines.top': False})





sns.countplot(df_2017['month_start'], ax=ax[0], palette="ch:.25")

ax[0].set_title('Count of Bike Rides per Month in 2017', fontsize=14)

ax[0].set_xticklabels(df_2017['month_start'].unique(),rotation=45)

ax[0].set_xlabel('Months')

ax[0].set_ylabel('Number of Rides')

ax[0].set_ylim(0, 300000) 





sns.countplot(df_2018['month_start'], ax=ax[1], palette="Blues")

ax[1].set_title('Count of Bike Rides per Month in 2018', fontsize=14)

ax[1].set_xticklabels(df_2018['month_start'].unique(),rotation=45)

ax[1].set_xlabel('Months')

ax[1].set_ylabel('Number of Rides')





plt.show()
plt.figure(figsize=(12,6))

red_patch = mpatches.Patch(color='b',label='2017')

blue_patch = mpatches.Patch( color='aliceblue',label='2018')



g=sns.countplot(df_2017['month_start'], palette="GnBu")

sns.countplot(df_2018['month_start'],palette="Blues",alpha=0.25)

g.set_title('Count of Bike Rides per Month in 2017 and 2018', fontsize=14)

g.set_xticklabels(df_2017['month_start'].unique(),rotation=45)

g.set_xlabel('Months')

g.set_ylabel('Count')

g.legend(handles=[blue_patch, red_patch])





plt.show()



plt.figure(figsize=(12,6))

red_patch = mpatches.Patch(color='r', label='2017')

blue_patch = mpatches.Patch(color='b', label='2018')



g=sns.countplot(df_2017['start_day_week'], palette="BrBG")

sns.countplot(df_2018['start_day_week'],color='r',alpha=0.25)

g.set_title('Count of Bike Rides by Weekday in 2017 and 2018', fontsize=14)

g.set_xticklabels(df_2017['start_day_week'].unique())

g.set_xlabel('Weekday')

g.set_ylabel('Count')

g.legend(handles=[blue_patch, red_patch])





plt.show()
trip_long=pd.pivot_table(df,index=['start_date'],values='trip_duration_seconds', aggfunc=np.mean )

trip_long.reset_index(inplace=True)

trip_long['start_day_week']=[x.weekday() for x in trip_long['start_date']]

trip_long['month_start']=[x.month for x in trip_long['start_date']]

month_dict={1:'January', 2:'February', 3:'March', 4:'April',5:'May', 6:'June', 7:'July', 8:'August',9:'September', 10:'October', 11:'November', 12:'December'}

trip_long['month_start']=trip_long['month_start'].map(month_dict)

day_voc={0:'Monday',1:'Tuesday',2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday',6:'Sunday' }



trip_long['start_day_week']=trip_long['start_day_week'].map(day_voc)

trip_long['month_start'].unique()

index_2017=[x.year==2017 for x in trip_long['start_date']]

trip_long['year']=trip_long['month_start']

trip_long['year'][index_2017]= '2017'

index_2018=[x.year==2018 for x in trip_long['start_date']]

trip_long['year'][index_2018]= '2018'





trip_long_17=trip_long[trip_long['year']=='2017']

trip_long_18=trip_long[trip_long['year']=='2018']
trip_long_17.rename(columns={'start_day_week':'Weekday'},inplace=True)



sns.set_style( {'axes.spines.left': False,

 'axes.spines.bottom': False,

 'axes.spines.right': False,

 'axes.spines.top': False})

plt.figure(figsize=(18,8))

g=sns.barplot(trip_long_17['month_start'],trip_long_17['trip_duration_seconds'], hue=trip_long_17['Weekday'],palette='BuGn',hue_order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

#sns.lineplot(trip_long_18['month_start'],trip_long_18['trip_duration_seconds'], hue=trip_long_18['start_day_week'])

g.set_title('Duration of Bikerides in Seconds', fontsize=16)

g.set(xlabel='Months', ylabel='Total Average Ride Duration')

print(df_2017.user_type.unique())

print(df_2018.user_type.unique())
#create dictionary of values to apply to the column

dict_type={'Member':'Annual Member', 'Casual':'Casual Member'}

df_2017.user_type=df_2017.user_type.apply(lambda x:dict_type[x] )
sns.set_style( {'axes.spines.left': False,

 'axes.spines.bottom': True,

 'axes.spines.right': False,

 'axes.spines.top': False})

fig, ax = plt.subplots(2,1, figsize=(10,6))

fig.tight_layout(pad=6.0)

sns.countplot(df_2017['month_start'], ax=ax[0],hue=df_2017['user_type'], palette="PiYG")

ax[0].set_title('Count of Bikerides in 2017 among Members and Casual Riders', fontsize=14)

ax[0].set_xticklabels(df_2017['month_start'].unique(), rotation=30)

ax[0].set_xlabel('')

ax[0].set_ylabel('Number of Rides')

ax[0].set_ylim(0, 200000) 

ax[0].legend(loc='upper left', frameon=False)







sns.countplot(df_2018['month_start'], ax=ax[1],hue=df_2018['user_type'], palette="BrBG")

ax[1].set_title('Count of Bikerides in 2018 among Members and Casual Riders', fontsize=14)

ax[1].set_xticklabels(df_2018['month_start'].unique(), rotation=30)

ax[1].set_xlabel('')

ax[1].set_ylabel('Number of Rides')

ax[1].legend(loc='upper left', frameon=False)

df['start_hour']=[int(str(x)[:2]) for x in df['start_time']]# to create a column just with the hour value without minutes

#groupby start hour and day to get the count of rides per hour each day

n=df[['start_hour','start_date','month_start']].groupby(['start_hour','start_date']).count()



n.reset_index(inplace=True)

n['start_day_week']=[x.weekday() for x in n['start_date']] #create column of weekdays in numbers

day_voc={0:'Monday',1:'Tuesday',2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday',6:'Sunday' }

#create column of word weekdays

n['start_day_week']=n['start_day_week'].map(day_voc)

#create column of year value

n['year']=[x.year for x in n['start_date']]

n['month']=[x.month for x in n['start_date']] #column with month in numbers

month_dict={1:'January', 2:'February', 3:'March', 4:'April',5:'May', 6:'June', 7:'July', 8:'August',9:'September', 10:'October', 11:'November', 12:'December'}

n['month_word']=n['month'].map(month_dict) #months in words



#rename columns to get better visual

n.rename(columns={'month_start':'count', 'start_day_week':'weekday', 'start_hour':'hour'}, inplace=True)
g = sns.FacetGrid(n, col="weekday", hue="month",

                  subplot_kws=dict(projection='polar'), height=4.5,

                  sharex=False, sharey=False, despine=False,col_order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])



# Draw a scatterplot onto each axes in the grid

g.map(sns.lineplot, "hour", "count",palette="PiYG")

g.set_titles("{col_name}") 

for i,v in enumerate(n['month_word'].unique()):

    sns.set_style('white')

    g=sns.FacetGrid(n[n['month']==(i+1)], col='weekday', col_wrap=7, height=1.5, aspect=2, col_order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

    g=g.map(sns.lineplot, 'hour','count','year', palette="PiYG")

    g.set(xlim=(0,24))

    g.set_xlabels('Time')

    g.set_ylabels('{}'.format(v))

    g.set_titles("{col_name}") 

    g.add_legend()

    g.despine(left=True)

    
n1=df[['start_day_week','month_start','year','user_type']].groupby(['start_day_week','month_start','year']).count()

n1.reset_index(inplace=True)

n1.rename(columns={'month_start':'month', 'start_day_week':'weekday', 'user_type':'count'}, inplace=True)

month_to_num={'January':0, 'February':1, 'March':2,'April':3,'May':4,'June':5,'July':6,'August':7,'September':8,'October':9,'November':10,'December':11}

weekday_to_num={'Monday':0, 'Tuesday':1, 'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}

n1['month1']=n1['month'].apply(lambda x: month_to_num[x])

n1['weekday1']=n1['weekday'].apply(lambda x: weekday_to_num[x])



num_to_month=['January', 'February','March','April','May','June','July','August','September','October','November','December']

num_to_weekday={0:'Monday', 1:'Tuesday', 2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}

n17=n1[n1['year']=='2017']

n18=n1[n1['year']=='2018']

n17p=n17.pivot(index='month1', columns='weekday1', values='count')

n18p=n18.pivot(index='month1', columns='weekday1', values='count')



cols=['Monday','Tuesday','Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']

n17p['month']=num_to_month

n17p.set_index('month',inplace=True)

n18p['month']=num_to_month

n18p.set_index('month',inplace=True)

n17p.columns=cols

n18p.columns=cols

sum17=np.sum(n17p,axis=1).tolist()

yearpermonth17=pd.DataFrame({'total_sum':sum17}).set_index(n17p.index)

sum18=np.sum(n18p,axis=1).tolist()

yearpermonth18=pd.DataFrame({'total_sum':sum18}).set_index(n18p.index)
percentagediff=np.round((n18p-n17p)/n18p,2)

percentagediff
g=sns.clustermap(percentagediff, cmap='mako', annot=percentagediff)

plt.figure(figsize=(12,6))

g=sns.heatmap(percentagediff,annot=True)

g.set_title('Count of Bike Rides per Month in 2018/2017', fontsize=14)

g.set_xticklabels(df_2017['month_start'].unique(),rotation=45)

#g.set_xlabel('')

g.set_ylabel('')





plt.figure(figsize=(12,6))

sns.set(style="whitegrid")

g=sns.scatterplot(yearpermonth17['total_sum'],yearpermonth17.index,palette="RdYlGn", marker='D',s=120)

sns.scatterplot(yearpermonth18['total_sum'],yearpermonth18.index,palette="PRGn",s=120)

g.set_title('Total Bike Rides per Month in 2017 and 2018', fontsize=14)

g.set_xlabel('')

g.set_ylabel('')

g.legend(loc='lower right', frameon=False)


