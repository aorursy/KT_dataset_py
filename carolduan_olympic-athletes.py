# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # matplotlib for additional customization
import seaborn as sns # Seaborn for plotting and styling
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import dataset
athlete = pd.read_csv('../input/athlete_events.csv')
# explore data
athlete.head()
# explore data
athlete.Medal.value_counts()
# feature engineering
# map the Medal 1 = Bronze, 2 = Silver, Gold = 3 (the best)
# map the Sex 1 = M, 2 = F
athlete.Medal = athlete.Medal.map({'Bronze':1, 'Silver':2, 'Gold':3})
athlete.Sex = athlete.Sex.map({'M':1, 'F': 0})
athlete.head()                                
# clean all NaN data (drop null values) - to only prize-winner
athlete.dropna(inplace=True)
athlete.Medal.value_counts()
# Is there any relationship between the medal and other numeric features of athlete?
# No, there is no strong correlations between olympic medal and athlete's physical features.
sns.heatmap(athlete.corr()[['Medal']], annot=True)
sns.set(rc={'figure.figsize':(11.7,8.27)})
# Distribution of the athlete's ages among prize-winners?
# The average age of Olympic prize-winners is 24 years old and the distribution is right-skewed
# A few of the athletes are more than 40 years old.
sns.distplot(athlete['Age'])
# US vs. China, which country performed better in terms of the Olympic medal numbers?
# According to the violinplot, the most frequent medal US and China got is the same - Silver.
# However, American got more Gold medal and less Bronze medal than China. So, US performed better in the Olympic history.
us_china = athlete.loc[(athlete.NOC=='USA')|(athlete.NOC=='CHN'),:]
sns.violinplot(us_china['NOC'], us_china['Medal'])
