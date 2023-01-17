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
data=pd.read_csv('../input/demographics-and-employment-in-the-united-states/CPSData.csv')
data.head()
data.info()
data.Industry.nunique()
data.Industry.value_counts()
data.State.value_counts()

data.Citizenship.value_counts()
#What proportion of interviewees are citizens of the United States?



(116639+7073)/131302
data.Race.value_counts()
pd.crosstab(data['Race'], columns=data['Hispanic'])

data.info()
pd.crosstab(data['Region'], columns=data['Married'])

pd.crosstab(data['Region'], columns=data['Married'].isna())

data.groupby(['Race', 'Hispanic']).size()

data.isnull().sum()

import missingno as msno 

msno.matrix(data) 

msno.bar(data) 
msno.heatmap(data) 

type(data.State.value_counts())
data.State.value_counts()
data.State[data['MetroAreaCode'].isnull()].value_counts()
data.State[~data['MetroAreaCode'].isnull()].value_counts()
data['MetroAreaCode'].isnull().value_counts()
data.State.value_counts()
data.Region[data.MetroAreaCode.isnull()].value_counts()

data.Region.value_counts()