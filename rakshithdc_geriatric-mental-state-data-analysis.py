

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualizing data

import seaborn as sns 

from collections import Counter

%matplotlib inline

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import plotly.figure_factory as ff

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import plotly.plotly as py

import plotly.graph_objs as go

import seaborn as sns
df = pd.read_csv('../input/Geriatrics.csv')

#shape of the dateset (column,rows)

df.shape
#description of dataset

# DM - diabetes , HTN - hypertension,

df.describe()
#first 5 items

#0 = female 1 = male

df.head(60)
#

def plot(group,column,plot):

    ax=plt.figure(figsize=(7,7))

    df.groupby(group)[column].sum().sort_values().plot(plot)

    

plot('Health','Gender','barh')
#Total patients count with normal/disease

explode = (0.2,0)  

fig1, ax1 = plt.subplots(figsize=(7,4))

ax1.pie(df['Health'].value_counts(), explode=explode,labels=['Normal','Disease'], autopct='%1.0f%%',

         startangle=180)

ax1.axis('equal')  

plt.legend()

plt.show()
#Total patients count with normal/disease

explode = (0.2,0)  

fig1, ax1 = plt.subplots(figsize=(7,4))

ax1.pie(df['Gender'].value_counts(), explode=explode,labels=['0','1'], autopct='%1.0f%%',

         startangle=180)

ax1.axis('equal')  

plt.legend()

plt.show()
fig1, ax1 = plt.subplots(figsize=(14,6))

sns.countplot(df['Gender'],hue=df['Age'])


fig1, ax1 = plt.subplots(figsize=(14,6))

sns.countplot(df['Personal_income'],hue=df['Age'])
fig1, ax1 = plt.subplots(figsize=(14,6))

sns.countplot(df['Education'],hue=df['Age'])
fig1, ax1 = plt.subplots(figsize=(14,6))

sns.countplot(df['Family_Type'],hue=df['Age'])
fig1, ax1 = plt.subplots(figsize=(14,6))

sns.countplot(df['Occupation'],hue=df['Age'])
fig1, ax1 = plt.subplots(figsize=(14,6))

sns.countplot(df['substance_abuse'],hue=df['Age'])
fig1, ax1 = plt.subplots(figsize=(14,6))

sns.countplot(df['Personal_income'],hue=df['Age'])
#duabetes

fig1, ax1 = plt.subplots(figsize=(14,6))

sns.countplot(df['DM'],hue=df['Age'])
#hypertension

fig1, ax1 = plt.subplots(figsize=(14,6))

sns.countplot(df['HTN'],hue=df['Age'])
fig1, ax1 = plt.subplots(figsize=(14,6))

sns.countplot(df['hearing_problem'],hue=df['Age'])
fig1, ax1 = plt.subplots(figsize=(14,6))

sns.countplot(df['vision_problem'],hue=df['Age'])
fig1, ax1 = plt.subplots(figsize=(14,6))

sns.countplot(df['mobility_problem'],hue=df['Age'])
fig1, ax1 = plt.subplots(figsize=(14,6))

sns.countplot(df['sleep_problem'],hue=df['Age'])