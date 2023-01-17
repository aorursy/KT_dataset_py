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
data=pd.read_csv('../input/chicago-city-crime-dataset/mvtWeek1.csv')
data.head()
data.tail()
data.info()
data.shape[0]
#How many rows of data (observations) are in this dataset?

#191641
#How many variables are in this dataset?

data.columns
data.describe()
#Using the "max" function, what is the maximum value of the variable "ID"?

#9181151
#What is the minimum value of the variable "Beat"?

#111



#How many observations have value TRUE in the Arrest variable (this is the number of crimes for which an arrest was made)?

#15536
data.Arrest.unique()
data.Arrest.value_counts()
#How many observations have a LocationDescription value of ALLEY?

#2308



data.LocationDescription.value_counts()
#data['Date1']=pd.to_datetime(data['Date'], format='%Y-%m-%d')
data['Date1'] = pd.to_datetime(data['Date'])

data['Date1'] = data['Date1'].dt.date
data
data[data.index==95821]
#What is the month and year of the median date in our dataset?

#2006-05-21

#May 2006
data['month'] = pd.to_datetime(data['Date1'])

data['year'] = pd.DatetimeIndex(data['Date1']).year
data
data['month'] = pd.DatetimeIndex(data['Date1']).month
data
data['weekday'] = pd.to_datetime(data['Date1']).apply(lambda x: x.weekday())

data
data['dayd'] = pd.to_datetime(data['Date'])

data['dayd'] = data['dayd'].dt.day
data
#https://medium.com/@swethalakshmanan14/simple-ways-to-extract-features-from-date-variable-using-python-60c33e3b0501
data['dayweek'] = pd.to_datetime(data['Date'])

data['dayweek'] = data['dayweek'].dt.day_name()
data
#On which weekday did the most motor vehicle thefts occur?

#on friday
data['dayweek'].value_counts()
#Each observation in the dataset represents a motor vehicle theft, 

#and the Arrest variable indicates whether an arrest was later made for this theft.

#Which month has the largest number of motor vehicle thefts for which an arrest was made?
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder() 

  
data['Arrest']= label_encoder.fit_transform(data['Arrest']) 

  
data[data['month']==12].describe()
data[data['month']==1].describe()
import matplotlib.pyplot as plt
data.info()

from pylab import rcParams

rcParams['figure.figsize'] = 10,8
#data['Date1'].value_counts().plot(kind='bar')
import seaborn as sns
sns.catplot(x="Date1", kind="count", palette="ch:.25", data=data);
data.boxplot(column='year',by='Arrest')
data[data['year']==2001].describe()
data[data['year']==2007].describe()
data[data['year']==2012].describe()
data['LocationDescription'].nunique()
data['LocationDescription'].value_counts()
t5 = ['STREET', 'PARKING LOT/GARAGE(NON.RESID.)', 'ALLEY', 'GAS STATION', 'DRIVEWAY - RESIDENTIAL' ]

top5 = data[data['LocationDescription'].isin(t5)].copy()

top5.shape
pd.crosstab(top5['LocationDescription'], columns=top5['Arrest'],normalize='index')

top5.loc[top5['LocationDescription']=='GAS STATION','weekday'].value_counts()

top5.loc[top5['LocationDescription']=='DRIVEWAY - RESIDENTIAL','weekday'].value_counts()
