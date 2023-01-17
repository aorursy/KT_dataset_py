import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt # visualizing data

import seaborn as sns 

from collections import Counter

%matplotlib inline

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import plotly.figure_factory as ff

import os

print(os.listdir("../input"))

import plotly.plotly as py

import plotly.graph_objs as go

import seaborn as sns

df = pd.read_csv('../input/BlackFriday.csv')

df.shape
#df.info()
df.describe()

df.describe()

df.head()
explode = (0.1,0)  

fig1, ax1 = plt.subplots(figsize=(12,7))

ax1.pie(df['Gender'].value_counts(), explode=explode,labels=['Male','Female'], autopct='%1.1f%%',

        shadow=True, startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.legend()

plt.show()
def plot(group,column,plot):

    ax=plt.figure(figsize=(12,6))

    df.groupby(group)[column].sum().sort_values().plot(plot)

    

plot('Gender','Purchase','bar')
fig1, ax1 = plt.subplots(figsize=(12,7))

sns.countplot(df['Age'],hue=df['Gender'])

plot('Age','Purchase','bar')
explode = (0.1, 0, 0)

fig1, ax1 = plt.subplots(figsize=(12,7))

ax1.pie(df['City_Category'].value_counts(),explode=explode, labels=df['City_Category'].unique(), autopct='%1.1f%%',

        shadow=True, startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.legend()

plt.show()
explode = (0.1, 0, 0)

fig1, ax1 = plt.subplots(figsize=(12,7))

ax1.pie(df.groupby('City_Category')['Purchase'].sum(),explode=explode, labels=df['City_Category'].unique(), autopct='%1.1f%%',

        shadow=True, startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.legend()

plt.show()
fig1, ax1 = plt.subplots(figsize=(12,7))

sns.countplot(df['City_Category'],hue=df['Age'])

#label=['Underage 0-17','Retired +55','Middleage 26-35','46-50 y/o','Oldman 51-55','Middleage+ 36-45','Youth']

explode = (0.1, 0)

fig1, ax1 = plt.subplots(figsize=(12,7))

ax1.pie(df['Marital_Status'].value_counts(),explode=explode, labels=['Yes','No'], autopct='%1.1f%%',

        shadow=True, startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.legend()

plt.show()
labels=['First Year','Second Year','Third Year','More Than Four Years','Geust']

explode = (0.1, 0.1,0,0,0)

fig1, ax1 = plt.subplots(figsize=(12,7))

ax1.pie(df.groupby('Stay_In_Current_City_Years')['Purchase'].sum(),explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.legend()

plt.show()
labels=['First Year','Second Year','Third Year','More Than Four Years','Geust']

#label=['Underage 0-17','Retired +55','Middleage 26-35','46-50 y/o','Oldman 51-55','Middleage+ 36-45','Youth']

explode = (0.1, 0.1,0,0,0)

fig1, ax1 = plt.subplots(figsize=(12,7))

ax1.pie(df['Stay_In_Current_City_Years'].value_counts(),explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.legend()

plt.show()
plot('Stay_In_Current_City_Years','Purchase','bar')
fig1, ax1 = plt.subplots(figsize=(12,7))

df['Occupation'].value_counts().sort_values().plot('bar')

plot('Occupation','Purchase','bar')


plot('Product_Category_1','Purchase','barh')


plot('Product_Category_2','Purchase','barh')
plot('Product_Category_3','Purchase','barh')
fig1, ax1 = plt.subplots(figsize=(12,7))

df.groupby('Product_ID')['Purchase'].count().nlargest(10).sort_values().plot('barh')
fig1, ax1 = plt.subplots(figsize=(12,7))

df.groupby('Product_ID')['Purchase'].sum().nlargest(10).sort_values().plot('barh')