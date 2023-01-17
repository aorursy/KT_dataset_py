import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

import pycountry

py.init_notebook_mode(connected=True)

import folium 

from folium import plugins



import pydicom



# Graphics in retina format 

%config InlineBackend.figure_format = 'retina' 



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

#plt.rcParams['image.cmap'] = 'viridis'



# palette of colors to be used for plots

colors = ["steelblue","dodgerblue","lightskyblue","powderblue","cyan","deepskyblue","cyan","darkturquoise","paleturquoise","turquoise"]



# Disable warnings in Anaconda

import warnings

warnings.filterwarnings('ignore')
basepath = '../input/osic-pulmonary-fibrosis-progression/'
train_info = pd.read_csv(basepath + 'train.csv')

train_info.head()
train_0 = train_info.loc[train_info.Weeks == 0]

train_0.head()
len(train_0)
fig, ax = plt.subplots(figsize=(15,5))

sns.countplot(train_0.Sex, palette="Reds_r", ax=ax);

ax.set_xlabel("")

ax.set_title("Gender counts in train on  the zero week ");
fig, ax = plt.subplots(figsize=(15,5))



sns.countplot(train_0.Age, color="orangered", ax=ax);

labels = ax.get_xticklabels();

ax.set_xticklabels(labels, rotation=90);

ax.set_xlabel("");

ax.set_title("Age distribution in train on the zero week");
fig, ax = plt.subplots(figsize=(15,5))



sns.countplot(train_0.SmokingStatus, color="orangered", ax=ax);

labels = ax.get_xticklabels();

ax.set_xticklabels(labels, rotation=90);

ax.set_xlabel("");

ax.set_title("Smoking status distribution in train on the zero week");
fig, ax = plt.subplots(2,1,figsize=(15,10))



sns.distplot(train_0.FVC, color="g", ax=ax[0]);

ax[0].set_xlabel("");

ax[0].set_title("Distribution of FVC in train on the zero week");



sns.distplot(train_0.Percent, color="r", ax=ax[1]);

ax[1].set_xlabel("");

ax[1].set_title("Distribution of Percent in train on the zero week");
percent_100 = train_0.FVC / train_0.Percent * 100

percent_100.mean()
train_0['Percent 100%'] = train_0.FVC / train_0.Percent * 100

train_group_sex = train_0.loc[:, ['FVC', 'Percent', 'Percent 100%','Sex']].groupby(['Sex']).mean()

train_group_sex
train_group_age = train_0.loc[:, ['FVC', 'Percent', 'Percent 100%','Age']].groupby(['Age']).mean()

train_group_age['Age'] = train_group_age.index

train_group_age.head()
fig, ax = plt.subplots(3,1,figsize=(15,17))



sns.regplot("Age", "FVC", data=train_group_age, truncate=False,

                  color="r", order=3, ax=ax[0])

ax[0].set_title("Distribution of average FVC by age in train on  the zero week");



sns.regplot("Age", "Percent", data=train_group_age, truncate=False,

                  color="g", order=3, ax=ax[1]);

ax[1].set_title("Distribution of average Percent by age in train on  the zero week");



sns.regplot("Age", "Percent 100%", data=train_group_age,truncate=False,

                  color="b", order=3, ax=ax[2])

ax[2].set_title("Distribution of average Percent 100% by age in train on  the zero week");
train_group_smoking = train_0.loc[:, ['FVC', 'Percent', 'Percent 100%','SmokingStatus']].groupby(['SmokingStatus']).mean()

train_group_smoking['SmokingStatus'] = train_group_smoking.index

train_group_smoking
fig, ax = plt.subplots(3,1,figsize=(15,17))



sns.barplot(data = train_group_smoking, x = 'SmokingStatus', y ="FVC", ax=ax[0])

ax[0].set_title("Distribution of average FVC by SmokingStatus in train on  the zero week");



sns.barplot(data = train_group_smoking, x = 'SmokingStatus', y ="Percent", ax=ax[1])

ax[1].set_title("Distribution of average Percent by SmokingStatus in train on  the zero week");



sns.barplot(data = train_group_smoking, x = 'SmokingStatus', y ="Percent 100%", ax=ax[2])

ax[2].set_title("Distribution of average Percent 100% by SmokingStatus in train on  the zero week");
def get_tuble(arr, ind):

    ans = np.array([])

    for element in arr:

        ans = np.append(ans, element[ind])

    return ans
train_group_smoking_sex = train_0.loc[:, ['FVC', 'Percent', 'Percent 100%','SmokingStatus', 'Sex']].groupby(['SmokingStatus', 'Sex']).mean()

train_group_smoking_sex['SmokingStatus'] = get_tuble(train_group_smoking_sex.index, 0)

train_group_smoking_sex['Sex'] = get_tuble(train_group_smoking_sex.index, 1)

train_group_smoking_sex
fig, ax = plt.subplots(3,1,figsize=(15,17))



sns.barplot(data = train_group_smoking_sex, x = 'SmokingStatus', y ="FVC", ax=ax[0], hue='Sex')

ax[0].set_title("Distribution of average FVC by SmokingStatus  and gender in train on  the zero week");



sns.barplot(data = train_group_smoking_sex, x = 'SmokingStatus', y ="Percent", ax=ax[1], hue='Sex')

ax[1].set_title("Distribution of average Percent by SmokingStatus and gender in train on  the zero week");



sns.barplot(data = train_group_smoking_sex, x = 'SmokingStatus', y ="Percent 100%", ax=ax[2], hue='Sex' )

ax[2].set_title("Distribution of average Percent 100% by SmokingStatus and gender in train on  the zero week");
train_group_smoking_age = train_0.loc[:, ['FVC', 'Percent', 'Percent 100%','SmokingStatus', 'Age']].groupby(['SmokingStatus', 'Age']).mean()

train_group_smoking_age['SmokingStatus'] = get_tuble(train_group_smoking_age.index, 0)

train_group_smoking_age['Age'] = get_tuble(train_group_smoking_age.index, 1)

train_group_smoking_age.head()
sns.pairplot(train_group_smoking_age)
train_0_sex = pd.get_dummies(train_0.Sex, prefix='Sex')

train_0_smoking= pd.get_dummies(train_0.SmokingStatus, prefix='SmokingStatus')

train_0 = pd.concat([train_0, train_0_sex, train_0_smoking], axis = 1)

train_0.head()
corrMatrix = train_0.iloc[:, 2:].corr()

sns.heatmap(corrMatrix, annot=True)