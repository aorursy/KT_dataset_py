# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/internet-privacy-poll/AnonymityPoll.csv')
data.head()
data.info()
data.Smartphone.value_counts()
data.Smartphone.isnull().value_counts()
pd.crosstab(data.Sex,columns=data.Region)
pd.crosstab(data.State,columns=data.Region)
pd.crosstab(data['Internet.Use'],columns=data.Smartphone)
data.isnull().sum()
limited = data[(data['Internet.Use'] == 1) | data.Smartphone == 1]

limited.head()
limited.info()
limited.isnull().sum()
data['Info.On.Internet'].mean()
data['Info.On.Internet'].value_counts()
data['Worry.About.Info'].value_counts()
(data['Worry.About.Info'].value_counts()[1])/data['Worry.About.Info'].count()
(data['Anonymity.Possible'].value_counts()[1])/data['Anonymity.Possible'].count()

 
(data['Tried.Masking.Identity'].value_counts()[1])/data['Tried.Masking.Identity'].count()

(limited['Privacy.Laws.Effective'].value_counts()[1])/limited['Privacy.Laws.Effective'].count()

import seaborn as sns

import matplotlib.pyplot as plt
age = limited.Age[~limited.Age.isnull()]  # remove the nulls so that I can draw histogram

sns.distplot(age, kde=True, bins=40)

plt.show()
limited[limited.Smartphone == 0]['Info.On.Internet'].describe()
# What proportion of smartphone users who answered the Tried.Masking.Identity question have tried

#  masking their identity when using the Internet?





maskers = limited[ (limited.Smartphone == 1) & (limited['Tried.Masking.Identity'] == 1)].shape[0]  # no. of obs

non_maskers = len(limited[ (limited.Smartphone == 1) & (limited['Tried.Masking.Identity'] == 0)] )



print(maskers / (maskers + non_maskers))



xx = limited[ (limited.Smartphone == 0) & (limited['Tried.Masking.Identity'] == 1) ].shape[0]

yy = limited[ (limited.Smartphone == 0) & (limited['Tried.Masking.Identity'] == 0)].shape[0]



print(xx / (xx + yy))