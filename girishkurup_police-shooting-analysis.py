# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
shooting=pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv')
shooting.info()
shooting.isnull()


import seaborn as sns
# to render the graphs
import matplotlib.pyplot as plt
# import module to set some ploting parameters
from matplotlib import rcParams


# This function makes the plot directly on browser
%matplotlib inline

# Seting a universal figure size 
rcParams['figure.figsize'] = 10,8
# let us find the missing values.represented as yellow lines
sns.heatmap(shooting.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# figure size
plt.figure(figsize=(12,5))

# using facetgrid that is a great way to get information of our dataset
g = sns.FacetGrid(shooting, col='gender',size=5)
g = g.map(sns.distplot, "age")
plt.show()
plt.figure(figsize=(12, 7))
sns.boxplot(x='gender',y='age',data=shooting,palette='winter')
plt.figure(figsize=(12, 7))
sns.boxplot(x='threat_level',y='age',data=shooting,palette='winter')

def impute_age(cols):
    age = cols[0]
    threat_level = cols[1]
    
    if pd.isnull(age):

        if threat_level == "attack":
            return 38

        elif threat_level == "other":
            return 38

        else:
            return 36

    else:
        return age
shooting['age'] = shooting[['age','threat_level']].apply(impute_age,axis=1)
shooting['gender'].fillna(shooting['gender'].mode()[0], inplace=True)
sns.countplot(x='armed',hue='manner_of_death',data=shooting,palette='rainbow')
shooting['armed'].mode()
shooting['armed'].fillna(shooting['armed'].mode()[0], inplace=True)
shooting['race'].value_counts()
sns.heatmap(shooting.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.subplot(2,1,1)
sns.countplot("race",data=shooting,hue="manner_of_death", palette="hls")
plt.ylabel("count", fontsize=18)
plt.xlabel("race", fontsize=18)
plt.title("race dist ", fontsize=20)
plt.show()
mostshotrace =pd.DataFrame(shooting.groupby('race')['state'].count())
mostshotrace  = mostshotrace .sort_values('state', ascending=False)
mostshotrace.head(30).plot(kind = "bar")
race_state3 = shooting.pivot_table(values='age', index='state', columns='race', fill_value=0)
race_state3
sns.set(style="ticks")
#exercise = sns.load_dataset("shooting")
g = sns.catplot(x="state", y="age", hue="race",height=10, data=shooting)
race_state = shooting.groupby(['state','race']).count()

race_state
race_state=race_state.reset_index()
race_state
race_state2 = race_state.pivot_table(values='age', index='state', columns='race', fill_value=0)
race_state2
def impute_race(cols):
    race = cols[0]
    state = cols[1]
    
    if pd.isnull(race):

        if state == "HI":
            return "O"

        elif state == "AZ":
            return "N"

        else:
            return "W"

    else:
        return race
shooting['race'] = shooting[['race','state']].apply(impute_race,axis=1)
#plt.subplot(2,1,1)
sns.set(style="ticks")
#exercise = sns.load_dataset("shooting")
g = sns.catplot(x="state", y="age", hue="manner_of_death",height=10, data=shooting)
plt.figure(figsize=(12, 7))
sns.boxplot(x='state',y='age',data=shooting,palette='winter')
sns.heatmap(shooting.isnull(),yticklabels=False,cbar=False,cmap='viridis')

df=shooting.drop(['flee'], axis=1)

df=shooting.drop(['id'],axis=1)
df.info()
##df_train['Estimated_Insects_Count'] = df_train['Estimated_Insects_Count'].astype("int16")

df['date']=pd.to_datetime(df['date'],dayfirst=True)
df['quarter'] = df['date'].dt.quarter
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['week_day'] = df['date'].dt.dayofweek
df.info()
df.drop(['name'],axis=1)
df=df.drop(['name'],axis=1)
df=df.drop(['date'],axis=1)
df.info()
df=df.drop(['flee'],axis=1)
df.info()
#df = pd.get_dummies(df, columns=["manner_of_death","armed","gender","race","city","state","signs_of_mental_illness","threat_level","body_camera"],\
                         #prefix=["death","arned","gender","race","city","st","ill","threat","photo"], drop_first=False)
#plt.show()
#plt.figure(figsize=(15,12))
#sns.heatmap(df.astype(float).corr(),vmax=1.0,  annot=True)
#plt.show()
df.info()
#df["manner_of_death"].value_counts()
#columns=["manner_of_death","armed","gender","race","city","state","signs_of_mental_illness","threat_level","body_camera"]
#for i in columns:
    #df[i].value.counts()
    
df["manner_of_death"].value_counts()
df["armed"].value_counts()
df["gender"].value_counts()
df["race"].value_counts()
df["city"].value_counts()
df["state"].value_counts()
df["signs_of_mental_illness"].value_counts()
df["threat_level"].value_counts()
df["body_camera"].value_counts()

from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
df["manner_of_death"]= encoder.fit_transform(df["manner_of_death"].fillna('Nan'))
df["armed"]= encoder.fit_transform(df["armed"].fillna('Nan'))
df["gender"]= encoder.fit_transform(df["gender"].fillna('Nan'))
df["race"]= encoder.fit_transform(df["race"].fillna('Nan'))
df["city"]= encoder.fit_transform(df["city"].fillna('Nan'))
df["state"]= encoder.fit_transform(df["state"].fillna('Nan'))
df["signs_of_mental_illness"]= encoder.fit_transform(df["signs_of_mental_illness"].fillna('Nan'))
df["threat_level"]= encoder.fit_transform(df["threat_level"].fillna('Nan'))
df["body_camera"]= encoder.fit_transform(df["body_camera"].fillna('Nan'))
df.info()
plt.show()
plt.figure(figsize=(15,12))
sns.heatmap(df.astype(float).corr(),vmax=1.0,  annot=True)
plt.show()
df.isnull().sum()

rateofshooting=df.groupby(['year','city']).count()
rateofshooting
rateofshooting=rateofshooting.reset_index()
rateofshooting
#Plotting the count of title by Crop damage or not category
sns.countplot(x='year', data=df, palette="hls",hue="quarter")
plt.xlabel("year", fontsize=16)
plt.ylabel("rate", fontsize=16)
plt.title("yearly rate if shooting", fontsize=20)
plt.xticks(rotation=45)
plt.show()
#Plotting the count of title by Crop damage or not category
sns.countplot(x='year',data=df, palette="hls",hue="week_day")
plt.xlabel("year", fontsize=16)
plt.ylabel("rate", fontsize=16)
plt.title("yearly rate if shooting", fontsize=20)
plt.xticks(rotation=45)
plt.show()
df1= shooting.pivot_table(values='age', index='race', columns='state', fill_value=0)
df1
sns.barplot(x="race", y="age", data=shooting)
plt.figure(figsize=(20, 12))

sns.countplot(x = 'state',
              data = shooting,
              order = shooting['state'].value_counts().index)
plt.show()
shooting.info()
df['state'].value_counts()[:10]
shootingystate=shooting['state'].value_counts()[:10]
shootingystate
shootingystate1=pd.DataFrame(shootingystate).reset_index()
import plotly.express as px
fig = px.pie(shootingystate1, values='state', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()