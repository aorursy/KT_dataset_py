# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



mental_health = pd.read_csv('../input/survey.csv')

mental_health.head()
# looks like we got a funny guy over here 

ax1 = mental_health['Age'].plot().set(xlabel='Index', ylabel='Age')
# so lets ged rid of obviously wrong ages

mental_health = mental_health[(mental_health['Age'] > 0) & (mental_health['Age'] < 100 )]

ax2 = mental_health['Age'].plot().set(xlabel='Index', ylabel='Age')
mental_health.describe()
# plot histogram of age variable

sns.distplot(mental_health['Age'], kde=False).set_title("Age Distribution")