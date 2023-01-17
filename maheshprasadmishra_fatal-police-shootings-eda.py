import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
ps_df = pd.read_csv("../input/data-police-shootings/fatal-police-shootings-data.csv")

ps_df.head(10)
ps_df.shape
ps_df.info()
ps_df.isnull().sum()
ps_df['gender'].value_counts()
ps_df.body_camera.replace({False:0,True:1},inplace = True)
ps_df.head()
ps_df['race'].value_counts()
100*ps_df.isnull().sum()/len(ps_df)
ps_df['armed'].value_counts().sort_values(ascending= True)
ps_df.groupby("armed").count()
ps_df.drop(['armed'],axis =1,inplace = True)
ps_df.head()
ps_df.drop(['id'],axis=1,inplace = True)
ps_df.head()
ps_df.signs_of_mental_illness.replace({False:0,True:1},inplace = True)
ps_df.head()
ps_df.groupby("threat_level").count()
ps_df['flee'].value_counts()
ps_df['flee'] = ps_df['flee'].replace(np.nan,'Not fleeing')
100*ps_df.isnull().sum()/len(ps_df)
ps_df['age'].value_counts()
100*ps_df.isnull().sum()/len(ps_df)
ps_df['gender'] = ps_df['gender'].replace(np.nan,'M')
100*ps_df.isnull().sum()/len(ps_df)
ps_df['race'].value_counts()
ps_df.groupby("race").count()
ps_df['race'] = ps_df['race'].replace(np.nan,'W')
ps_df.replace(to_replace = ['A'], value = ['Asian'], inplace = True)

ps_df.replace(to_replace = ['B'], value = ['Black Non-Hispanic'], inplace = True)

ps_df.replace(to_replace = ['H'], value = ['Hispanic'], inplace = True)

ps_df.replace(to_replace = ['N'], value = ['Native American'], inplace = True)

ps_df.replace(to_replace = ['O'], value = ['Other'], inplace = True)

ps_df.replace(to_replace = ['W'], value = ['White Non-Hispanic'], inplace = True)
ps_df.head()
100*ps_df.isnull().sum()/len(ps_df)
ps_df['age'].median()
ps_df.loc[:,"age"]=ps_df["age"].fillna(ps_df["age"].median())
100*ps_df.isnull().sum()/len(ps_df)
bins = [0,16,32,48,64,200]

labels = ['0-16', '16-32', '32-48', '48-64', '64+']

ps_df['agegroup']=pd.cut(ps_df['age'], bins, labels = labels)
ps_df.head()
ps_df.drop(['age'],axis=1,inplace=True)
ps_df['agegroup'].value_counts()
## check for null values for final time

100*ps_df.isnull().sum()/len(ps_df)
#check for Dataset

ps_df.head()
ps_df['date'] = pd.to_datetime(ps_df['date'])

ps_df['year']= ps_df['date'].dt.year

ps_df['month']= ps_df['date'].dt.month

ps_df['day']= ps_df['date'].dt.day
ps_df.head()
ps_df.drop(['date','month','day'],axis =1, inplace = True)
ps_df.head()
plt.figure(figsize = (10, 5))

sns.countplot('race',data = ps_df)

plt.figure(figsize = (10, 5))

sns.countplot('year',data = ps_df)
ps_df['year'].value_counts()
ps_df['year'].value_counts(normalize=True) * 100
plt.figure(figsize = (10, 5))

sns.countplot('agegroup',data = ps_df)
plt.figure(figsize = (10, 5))

sns.countplot('manner_of_death',data = ps_df)
ps_df.select_dtypes(include='object')
## Dropping the name column

ps_df.drop(['name'],axis = 1,inplace = True)
ps_df.head()
ps_df.state.unique()
plt.figure(figsize = (25, 10))

sns.countplot('state',data = ps_df)
sns.countplot('flee',data = ps_df)
sns.countplot('threat_level',data = ps_df)
sns.countplot('body_camera',data = ps_df)
plt.figure(figsize = (15, 10))

sns. countplot('race',data = ps_df, hue = 'gender')
plt.figure(figsize = (15, 10))

sns. countplot('race',data = ps_df, hue = 'agegroup')
plt.figure(figsize = (15, 10))

sns. countplot('agegroup',data = ps_df, hue = 'race')
pd.crosstab(ps_df.state,ps_df.race,normalize = False)