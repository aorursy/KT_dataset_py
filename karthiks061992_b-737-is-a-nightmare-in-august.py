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
#Iam a beginner to datascience and this is my first dataset..tried my best to get something of it
dataset =pd.read_csv('/kaggle/input/wildlife-strikes/database.csv')#reading the dataset
dataset.columns

features=['Incident Year','Incident Month','Incident Year','Record ID','Species Name','Aircraft']
dataset=dataset[features]
dataset.head()

dataset['Incident Year'].isnull().sum()#no null values in incident years
#46822 null values in Engine Type(dropping)
dataset.dropna(inplace=True)
dataset.isnull().sum()
dataset.columns
dataset['Incident Month'].isnull().sum()
dataset['Incident Month'].value_counts()
#clearly august month has the most number of cases 
import seaborn as sn
dataset.groupby('Incident Month')['Record ID'].count().plot.line()

#August is the worst month for planes to hit

dataset.columns

dataset['Aircraft'].value_counts()
#replacing the unknown aicraft with secret aircraft as most countries label them as unidentified
dataset['Aircraft']=dataset['Aircraft'].str.replace('UNKNOWN','secret')
dataset['Aircraft'].value_counts()
dataset.groupby('Aircraft')['Species Name'].count()
#secret flights are highest to get hit (maybe considered as low protection or flights that serve in remote areas)
dataset.groupby(['Species Name','Aircraft'])['Aircraft'].count().sort_values(ascending=False)
import seaborn as sn
dataset.groupby(['Species Name','Aircraft'])['Aircraft'].count().sort_values(ascending=False)

#from the above analysis we can come to a conclusion that august is the worst month for getting hit and most of the military and other secret aircrafts get hit 
#by Mourning doves and in the era of passenger flights B 737 300 gets hit a lot of times by medium sized birds 