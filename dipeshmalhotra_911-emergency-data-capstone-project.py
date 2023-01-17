import numpy as np

import pandas as pd
%matplotlib inline

import matplotlib.pyplot as plot

import seaborn as sns

sns.set_style('whitegrid')

plot.rcParams["figure.figsize"]=(30,15)

SMALL_SIZE = 20

MEDIUM_SIZE = 22

BIG_SIZE = 25



plot.rc('font', size=BIG_SIZE)          # controls default text sizes

plot.rc('axes', titlesize=BIG_SIZE)     # fontsize of the axes title

plot.rc('axes', labelsize=BIG_SIZE)    # fontsize of the x and y labels

plot.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels

plot.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels

plot.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize

plot.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title
df=pd.read_csv("../input/montcoalert/911.csv")
df.info()
df.head()
df['zip'].value_counts().head(5)
df['twp'].value_counts().head(5)
df['addr'].value_counts().head(5)
def extract(Title):

   new=Title.split(':',1)

   return new[0]



df['Dept']=df.apply(lambda x:extract(x["title"]),axis=1)
def extract2(Title):

   new=Title.split(':',1)

   return new[1]



df['ExactR']=df.apply(lambda x:extract(x["title"]),axis=1)
fig,ax = plot.subplots()

b=sns.countplot(df['Dept'],data=df,palette='Set1',ax=ax)

b.axes.set_title("Dept vs Number of calls")

b.set_xlabel("Department")

b.set_ylabel("Number of calls")

plot.savefig('noofcalls.jpg', format='jpeg', dpi=70)

fig,ax = plot.subplots()

b=sns.countplot(x=df['ExactR'],palette='Set1',data=df,order=df.ExactR.value_counts().iloc[:5].index,ax=ax)

b.axes.set_title("Reasons vs Number of calls")

b.set_xlabel("Reason")

b.set_ylabel("Number of calls")

plot.savefig('noofcallsbyreason.jpg', format='jpeg', dpi=70)
df['timeStamp'].astype('str')
df['timeStamp']=pd.to_datetime(df['timeStamp'])

df['timeStamp']
df['Hour']=pd.Series(df['timeStamp'].apply(lambda time: time.hour))

df['Month']=pd.Series(df['timeStamp'].apply(lambda time: time.month))

df['Day']=pd.Series(df['timeStamp'].apply(lambda time: time.day_name()))
df.info()
df['Day'] = df['Day'].map({'Monday': 'Mon', 'Tuesday': 'Tue','Wednesday': 'Wed','Thursday': 'Thu','Friday': 'Fri','Saturday': 'Sat','Sunday': 'Sun'}).astype(str)
fig,ax=plot.subplots()

sns.countplot(df['Day'],hue=df['Dept'],palette='Set1',order=df['Day'].value_counts().index,ax=ax) # Checking for the weekday with most calls

plot.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0)
fig,ax=plot.subplots(ncols=3)

sns.countplot(df['Day'],data=df,palette='Set1',order = df['Day'].value_counts().index, ax=ax[0]).set_title('All seasons')

sns.countplot(df['Day'][df['Month'].isin([10,11,12,1,2,3])],palette='Set1',order = df['Day'].value_counts().index,ax=ax[1]).set_title('Winter')

sns.countplot(df['Day'][df['Month'].isin([4,5,6,7,8,9])],palette='Set1',order = df['Day'].value_counts().index,ax=ax[2]).set_title('Summer')
fig,ax=plot.subplots()

sns.countplot(df['Month'],hue=df['Dept'],palette='Set1',ax=ax)

plot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0)
bymonth=df.groupby('Month')['Dept'].count() # Used only one columns to save space and reduce redundancy

bymonth
bymonth.plot()
sns.lmplot(x='Month',y='Dept',data=bymonth.reset_index(),height=8,aspect=2)
df['Date']=pd.Series(df['timeStamp'].apply(lambda x:x.date()))
df.groupby('Date').count()['title'].plot()

plot.title('All calls to 911')
df[df['Dept']=='Traffic'].groupby('Date').count()['Dept'].plot()

plot.title('911 calls due to traffic')
df[df['Dept']=='Fire'].groupby('Date').count()['Dept'].plot()

plot.title('911 calls due to Fire')
df[df['Dept']=='EMS'].groupby('Date').count()['Dept'].plot()

plot.title('911 calls to Emergency Medical Services')
df.groupby(by=['Day','Hour']).count()['Dept']
de=df.groupby(by=['Day','Hour']).count()['Dept'].unstack()

de.head()


sns.heatmap(de,cmap='summer_r')
sns.clustermap(de,cmap='summer_r')
dm=df.groupby(by=['Day','Month']).count()['Dept'].unstack()

dm.head()
sns.heatmap(dm,cmap='summer_r')
sns.clustermap(dm,cmap='summer_r')