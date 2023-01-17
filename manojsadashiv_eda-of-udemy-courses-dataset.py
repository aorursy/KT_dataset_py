import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime as dt
warnings.simplefilter(action='ignore')
sns.set(style="ticks", color_codes=True,palette='inferno_r')
udemy=pd.read_csv('../input/udemy-courses/udemy_courses.csv')
udemy.head()
udemy.info()
cat=udemy.select_dtypes(include=['object','bool']).columns
cont=udemy.select_dtypes(exclude=['object','bool']).columns
cont
udemy1=udemy.copy()
mask = udemy1.applymap(type) != bool
d = {True: 'TRUE', False: 'FALSE'}

udemy1 = udemy1.where(mask, udemy1.replace(d))
plt.figure(figsize=(8,8))
sns.heatmap(udemy.corr(),cmap='YlGnBu',annot=True,square=True)
plt.xticks(rotation=45)
plt.tight_layout()
#coolwarm YlGnBu 
g=sns.pairplot(udemy1,diag_kind='kde',hue='is_paid',palette='husl')
def col_types():
    print(cat)
    print(cont)
col_types()
udemy.groupby('is_paid').is_paid.count()
udemy.groupby('level').is_paid.value_counts()
sns.countplot(x='level',hue='is_paid',data=udemy)
plt.xticks(rotation=8)
plt.show()
udemy.groupby('subject').is_paid.value_counts()
sns.countplot(x='subject',hue='is_paid',data=udemy)
plt.xticks(rotation=8)
plt.show()
udemy.describe()
col_types()
sns.catplot(x='level',y='price',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=10)
#udemy[(udemy['price']==200) & (udemy['level']=='All Levels')]

sns.catplot(x='level',y='num_lectures',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=10)
udemy.loc[udemy['num_lectures']>400]['course_title']
udemy.loc[udemy['num_lectures']>750]
sns.catplot(x='level',y='num_reviews',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=10)
a=udemy.loc[udemy['num_reviews']>5000]
a.groupby('subject').is_paid.value_counts()
a.groupby('level').subject.value_counts()
col_types()
sns.catplot(x='level',y='num_subscribers',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=10)
udemy.loc[udemy['num_subscribers']>250000]
udemy.loc[(udemy['num_subscribers']>150000) &(udemy['level']=='Beginner Level') ]
col_types()
sns.catplot(x='level',y='content_duration',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=10)
udemy.loc[udemy['content_duration']==0]
udemy1=udemy.copy()
cond=udemy['content_duration']==0
udemy1.drop(cond.index,axis=0,inplace=True)
udemy1.loc[udemy1['content_duration']==0]
cond=udemy['content_duration']==0
udemy.drop(udemy[cond].index,axis=0,inplace=True)
udemy[udemy['content_duration']==0]
col_types()
sns.catplot(x='subject',y='price',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=8)
sns.catplot(x='subject',y='num_subscribers',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=10)
sns.catplot(x='subject',y='num_reviews',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=8)
udemy.loc[udemy['num_reviews']>25000]
sns.catplot(x='subject',y='num_lectures',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=8)
udemy.loc[udemy['num_lectures']>700]
sns.catplot(x='subject',y='content_duration',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=8)
udemy.loc[udemy['content_duration']>50][['course_title','subject']]
col_types()
udemy0=udemy.copy()
udemy0['published_timestamp'] = pd.to_datetime(udemy['published_timestamp'])
udemy0['published_date'] = udemy0['published_timestamp'].dt.date
udemy0['published_year'] = pd.DatetimeIndex(udemy0['published_date']).year
udemy0.groupby(['published_year']).count()
plt.figure(figsize = (9,4))
sns.countplot(data = udemy0, x = 'published_year')
plt.show()
udemy0.nlargest(5, 'published_timestamp')
