import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (6, 4)

%matplotlib inline



color = sns.color_palette()

import warnings

warnings.filterwarnings('ignore') 
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample = pd.read_csv('../input/sample_submission.csv')
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])

train['pickup_hour'] = train.pickup_datetime.dt.hour

train['pickup_week'] = train.pickup_datetime.dt.weekday

train['pickup_month'] = train.pickup_datetime.dt.month
whathap=train[train['pickup_month']==3]

whathap=whathap[whathap['pickup_week']==6]

whathap=whathap[whathap['pickup_hour']==1]

#print(whathap.head(200))

print(whathap.ix[10685])

print(whathap.ix[12559])

train.iloc[646333].T