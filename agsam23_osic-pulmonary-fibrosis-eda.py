import pandas as pd

import numpy as np



import matplotlib

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



import seaborn as sns

import plotly.express as px



import os

import random

import re

import math

import time





import warnings

warnings.filterwarnings("ignore")





seed_val = 42

random.seed(seed_val)

np.random.seed(seed_val)



black_red = [

    '#1A1A1D', '#4E4E50', '#C5C6C7', '#6F2232', '#950740', '#C3073F'

]

plt.style.use('fivethirtyeight')
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

sample = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
print(train.head())

print(len(train))
print('Train features:{}'.format(train.columns.tolist()))

print('Test features:{}'.format(test.columns.tolist()))
print(test.shape)

print(train.isnull().values.any())

print(test.isnull().values.any())
data = train.groupby(by="Patient")[["Patient", "Age", "Sex", "SmokingStatus"]].first().reset_index(drop=True)

fig = plt.figure(constrained_layout = True, figsize = (20,9))



#create grid



grid = gridspec.GridSpec(ncols = 4, nrows = 2, figure = fig)



ax1 = fig.add_subplot(grid[0, :2])

ax1.set_title('Gender Distribution')





sns.countplot(data['Sex'],

             alpha = 0.9,

             ax = ax1,

             color = '#C3073F',

             label = 'Train',

             order=train['Sex'].value_counts().index)



ax1.legend()



ax2 = fig.add_subplot(grid[0, 2:])

ax2.set_title('Smoking Status')

sns.countplot(data['SmokingStatus'],

             alpha = 0.9,

             ax = ax2, 

             color = '#C3073F',

             label = 'Train')

ax2.legend()

plt.xticks(rotation = 20)



ax3 = fig.add_subplot(grid[1, :])

ax3.set_title('Age Distribution')

sns.distplot(data['Age'], ax = ax3, label ='Train', color = '#C3073F')

ax3.legend()

plt.show()
fig = plt.figure(constrained_layout = True, figsize = (20,9))



#create grid



grid = gridspec.GridSpec(ncols = 4, nrows = 2, figure = fig)



ax1 = fig.add_subplot(grid[0, :2])

ax1.set_title('Gender Distribution')



sns.countplot(test.Sex,

             alpha = 0.9,

             ax = ax1,

             color = '#1A1A1D',

             label = 'Test',

             order=train['Sex'].value_counts().index)



ax1.legend()



ax2 = fig.add_subplot(grid[0, 2:])

ax2.set_title('Smoking Status')

sns.countplot(test.SmokingStatus,

             alpha = 0.9,

             ax = ax2, 

             color = '#1A1A1D',

             label = 'Test',

             order=train['SmokingStatus'].value_counts().index)

ax2.legend()

plt.xticks(rotation = 20)



ax3 = fig.add_subplot(grid[1, :])

ax3.set_title('Age Distribution')

sns.distplot(test.Age, ax = ax3, label ='Test', color = '#1A1A1D')

ax3.legend()

plt.show()


data = train.groupby(by="Patient")["Weeks"].count().reset_index(drop=False)

data = data.sort_values(['Weeks']).reset_index(drop=True)

plt.figure(figsize = (12, 6))

sns.barplot(x=data["Patient"], y=data["Weeks"], color='#950740')



plt.title("Number of Entries per Patient", fontsize = 17)

plt.xlabel('Patient', fontsize=14)

plt.ylabel('Frequency', fontsize=14)

plt.show()
fig = plt.figure(constrained_layout = True, figsize = (15,9))



#create grid



grid = gridspec.GridSpec(ncols = 1, nrows = 3, figure = fig)

ax1 = fig.add_subplot(grid[0, :])



sns.distplot(train.FVC, ax = ax1, color = '#950740')

ax1.set_title('FVC Distribution')



ax2 = fig.add_subplot(grid[1, :])

sns.distplot(train.Percent, ax = ax2, color = '#C3073F')

ax2.set_title('Percent Distribution')





ax3 = fig.add_subplot(grid[2, :])

sns.distplot(train.Weeks, ax = ax3, color = '#6F2232')

ax3.set_title('Number of weeks before/after CT scan')

black_red1 = [

    '#1A1A1D', '#950740', '#C3073F'

]

fig = px.sunburst(data_frame = train,

                 path = [ 'SmokingStatus','Sex','Age'],

                 color = 'Sex',

                 color_discrete_sequence = black_red1,

                 maxdepth = -1,

                 title = 'Sunburst Chart SmokingStatus > Gender > Age')

fig.update_traces(textinfo = 'label+percent parent')

fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

fig.show()
fig = plt.figure(constrained_layout = True, figsize = (12,6))



#create grid



grid = gridspec.GridSpec(ncols = 4, nrows = 1, figure = fig)



ax1 = fig.add_subplot(grid[0, :2])



sns.barplot(x = train.SmokingStatus, y = train.Percent, palette = black_red1, ax = ax1)

ax1.set_title('Percent per Smoking Status')

plt.xticks(rotation=30)





ax2 = fig.add_subplot(grid[0, 2:])

sns.barplot(x = train.SmokingStatus, y = train.FVC, palette = black_red1, ax = ax2)

ax2.set_title('FVC per Smoking Status')

plt.xticks(rotation=40)

plt.show()
fig = plt.figure(constrained_layout = True, figsize = (20,6))



#create grid



grid = gridspec.GridSpec(ncols = 6, nrows = 1, figure = fig)



ax1 = fig.add_subplot(grid[0, :2])



sns.scatterplot(x=train['FVC'] , y =train['Age'], palette = [black_red[1], black_red[4]], ax = ax1, hue = train['Sex'],

               style = train['Sex'])

ax1.set_title('Correlation between FVC and Age')

ax1.legend(loc = 'lower right')



ax2 = fig.add_subplot(grid[0, 2:4])

sns.scatterplot(x='FVC', y='Percent', data = train, palette = [black_red[1], black_red[4]], ax = ax2, hue = train['Sex'],

               style = train['Sex'])

ax2.set_title('Correlation between FVC and Percent')

ax2.legend(loc = 'upper right')



ax3 = fig.add_subplot(grid[0, 4:6])

sns.scatterplot(x='Percent', y='Age', data = train, palette = [black_red[1], black_red[4]], ax = ax3, hue = train['Sex'],

               style = train['Sex'])

ax3.set_title('Correlation between Percent and Age')

ax3.legend(loc = 'upper right')

plt.show()