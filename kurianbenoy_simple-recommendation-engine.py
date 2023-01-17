# analysing the Dataset

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from IPython.core.interactiveshell import InteractiveShell 

InteractiveShell.ast_node_interactivity = "all"
import pandas as pd

import pandas_profiling 
# Reading the data-file

df = pd.read_csv('/kaggle/input/location-visited-by-travller-in-india-fake/dataset.csv')
df.head()
df.profile_report()
# no of unique users in our dataset

df['Name'].nunique()
# no of unique city

df['City'].nunique()
# no of unique country

df['Country'].unique()
import seaborn as sns
# Analysing user preference w.r.t gender

sns.countplot(x='prefeerences', hue='gender', data=df)
# counting no of cities

sns.countplot(x='City', data=df)
# installing recommendation engine python package

! pip install turicreate
# read dataset

import turicreate as tc

actions = tc.SFrame.read_csv('/kaggle/input/location-visited-by-travller-in-india-fake/dataset.csv')
# Create a recommendation engine for a Particular user with name and engine

model = tc.recommender.create(actions, 'Name', 'City')
## Recommended cities

model.recommend()