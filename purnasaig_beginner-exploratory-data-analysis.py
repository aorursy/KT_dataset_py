import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#loading data into variable

data = pd.read_csv('../input/MetObjects.csv')
data.head()#shows first 5 rows
data.info()
data.shape
data.columns
data.describe()
data.isna().sum()
#view the missing values in ascending order

data.isna().sum().sort_values(ascending = False)
#lets view the columns that have more than 50% missing/NULL values.

data.isna().sum().sort_values(ascending = False).head(26)
data.isna().sum().sort_values(ascending = False).tail(17)
data['Department'].value_counts()
fig= plt.figure(figsize=(18, 9))

sns.countplot(data['Department'])
data['Department'].value_counts()[:20].plot(kind='barh')
data['Object ID'].value_counts().sum()
data['Is Public Domain'].value_counts()
data['Repository'].value_counts()[:20].plot(kind='barh')
data['Object Name'].value_counts()[:10].plot(kind = 'barh')
data['Culture'].value_counts()[:10].plot(kind = 'bar')
data['Culture'].value_counts()[:10].plot(kind = 'area')
data['Culture'].value_counts()[:10].plot(kind = 'pie')
data['Artist Role'].value_counts()[:10].plot(kind = 'pie')
data['Artist Display Name'].value_counts().head(5).plot(kind = 'bar')
data['Country'].value_counts().head(15)
data.Country.value_counts()[:10]
data['Country'].value_counts()[:20].plot(kind='barh')
# Let's check the correlation between the variables 

plt.figure(figsize=(20,10)) 

sns.heatmap(data.corr(), annot=True)