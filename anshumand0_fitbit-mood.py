##############################################

# Imports

##############################################



# System

import glob

import io

import os



# Data analysis

import math

import pandas as pd

import numpy as np



# Machine learning

import sklearn

import tensorflow.compat.v1 as tf

from tensorflow.python.data import Dataset 

import xgboost



# Visualization

from matplotlib import pyplot as plt

import seaborn as sns

import eli5

from pdpbox import pdp

import shap



# Other

import re

from collections import OrderedDict

from IPython.display import display
##############################################

# Initial Setup

##############################################



tf.disable_v2_behavior()

pd.options.display.max_rows = 10

pd.options.display.float_format = '{:.1f}'.format

np.random.seed(1)
##############################################

# Import Data

##############################################



# Import data sets

df = pd.read_csv(io.open("/kaggle/input/fitbitmood/Fitbit-Mood/daylio_export_2020_08_21.csv", "r"), sep=",", index_col=None)



# Reorder indices

df = df.reindex(np.random.permutation(df.index))
##############################################

# Data Information

##############################################



"""

Prints basic information about the dataframe including columns, head, tail, info, describe, and null values



Args: df (dataframe)

Return: null

"""

def quickSummary(df):

    print('\n========== Columns ==========\n')

    print(df.columns.values)

    print('\n========== Numeric Columns ==========\n')

    x = (df.dtypes != 'object')

    print(list(x[x].index))

    print('\n========== Non-Numeric Columns ==========\n')

    x = (df.dtypes == 'object')

    print(list(x[x].index))

    print('\n========== Head ==========\n')

    print(df.head(10))

    print('\n========== Tail ==========\n')

    print(df.tail(10))

    print('\n========== Info ==========\n')

    print(df.info())

    print('\n========== Describe ==========\n')

    print(df.describe())

    print('\n========== Null Values ==========\n')

    print(df.isnull().sum())

    

quickSummary(df)
##############################################

# Clean Data

##############################################



# Convert 'mood' to categorical integers

df['mood'] = df['mood'].map({'awful': 1, 'bad': 2, 'meh': 3, 'good': 4, 'rad': 5})



# Convert 'mood' to categorical integers

df['weekday'] = df['weekday'].map({'Sunday': 1, 'Monday': 2, 'Tuesday': 3, 'Wednesday': 4, 'Thursday': 5, 'Friday': 6, 'Saturday': 7})



# Convert 'date' to datetime format

df['date'] = pd.to_datetime(df['full_date'] + ' ' + df['time'])



# Convert 'note' from string to 'has_note' boolean

df['has_note'] = df['note'].notnull() - 0



# Convert 'activities' to one-hot encoded variables

df['activities'] = df['activities'].fillna('DROP')

df_activities = df['activities'].str.split(' | ', expand=True).stack().str.get_dummies().sum(level=0).add_prefix('activity_')

df_activities.rename(columns={'activity_Computer':'activity_computer'}, inplace=True)



quickSummary(df_activities)
# Drop activities that were used in < X% of entries

for col in df_activities:

    print(col + ': ' + str(df_activities[col].sum()) + ' (' + str(round(df_activities[col].sum()/df.shape[0]*100,2)) + '%)')



df_activities = df_activities.drop([col for col in df_activities if df_activities[col].sum()/df.shape[0]*100 < 3], axis=1)
# Concatenate activities to original df and drop unnecessary columns

df = pd.concat([df, df_activities], axis=1)

df = df.drop(['full_date', 'time', 'note', 'activities', 'activity_DROP'], axis=1)





quickSummary(df)
print("My average mood score is: " + str(df['mood'].mean()))

sns.distplot(df['mood'])
sns.barplot(x=df['weekday'],y=df['mood'])
df.loc[df['mood'] == 3].sum()
