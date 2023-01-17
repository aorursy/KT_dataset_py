!pip install plotly_express
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly_express as px

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df= pd.read_csv('../input/train.csv')

train_df.head()
test_df = pd.read_csv('../input/test.csv')

test_df.head()
(train_df.shape, test_df.shape)
gender_sub_df = pd.read_csv('../input/gender_submission.csv')

gender_sub_df.head()
gender_sub_df.shape
# datatype of columns

train_df.info()
# find missing data

sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
import warnings

warnings.filterwarnings("ignore")
sns.pairplot(train_df[['Survived','Pclass','Age','Fare']], dropna=True)
plot1 = px.histogram(train_df, x='Age', color='Pclass')

plot1
px.histogram(train_df, x='Age', color='Survived', facet_col='Pclass')
px.histogram(train_df, x='Fare', color='Pclass')
px.histogram(train_df, x='Fare', facet_col='Pclass', color='Survived', range_x=[0,200], nbins=50)
px.histogram(train_df, x='Age', color='Sex')
px.histogram(train_df, x='Age', color='Sex', facet_col='Survived')
px.histogram(train_df, x='Age', color='Sex', facet_col='Survived', facet_row='Pclass')
sns.countplot(x='Survived', hue='Pclass', data=train_df)
sns.heatmap(train_df[['Survived','Pclass']])