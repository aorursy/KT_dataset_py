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
import numpy as np

import pandas as pd

from pandas import DataFrame,Series

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

import warnings

warnings.filterwarnings('ignore')
data= pd.read_excel("/kaggle/input/lung-capacity/LungCap.xls")

data.head()
data.tail()
data.info()
data['Gender'].value_counts()
data['Gender'].value_counts(normalize=True)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

sns.countplot('Caesarean',data=data,ax=axes[0,0])

sns.countplot('Smoke',data=data,ax=axes[0,1])

sns.countplot('Gender',data=data,ax=axes[0,2])

sns.distplot(data['Height(inches)'],ax=axes[1,0])

sns.distplot(data['LungCap(cc)'],kde=False,ax=axes[1,1])

sns.distplot(data['Age( years)'],kde=True,ax=axes[1,2])

plt.show()
fig,axes=plt.subplots(1,3,figsize=(15,5))

data['Gender'].value_counts().plot.pie(explode = [0,0],autopct = '%1.0f%%',ax=axes[0],shadow = True)

data['Smoke'].value_counts().plot.pie(explode = [0.2,0],autopct = '%1.0f%%',ax=axes[1],shadow = True)

data['Caesarean'].value_counts().plot.pie(explode = [0.1,0],autopct = '%1.0f%%',ax=axes[2],shadow = True)

plt.show()
fig=plt.figure(figsize=(15,6))

sns.countplot(x='Gender',data=data,hue='Age( years)',palette='Accent')

plt.title('GENDER vs AGE')

plt.show()
fig,axes=plt.subplots(1,2,figsize=(15,6))

sns.boxplot(y='Age( years)',data=data,x='Caesarean',ax=axes[0])

axes[0].set_title('CAESAREAN vs AGE',color='darkblue')

sns.boxplot(y='LungCap(cc)',data=data,x='Gender',ax=axes[1])

axes[1].set_title('GENDER vs LUNG CAPACITY',color='indigo')

plt.show()
fig,axes=plt.subplots(1,2,figsize=(15,6))

sns.violinplot(x='Smoke',data=data,y='LungCap(cc)',ax=axes[0])

sns.violinplot(x='Smoke',data=data,y='Height(inches)',ax=axes[1])

axes[0].set_title('SMOKE vs LUNG CAPACITY',color='indigo')

axes[1].set_title('SMOKE vs HEIGHT',color='darkgreen')

plt.show()
fig,ax=plt.subplots(1,2,figsize=(15,6))

sns.boxplot(x='Smoke',data=data,y='Age( years)',ax=ax[0])

sns.violinplot(x='Gender',data=data,y='Height(inches)',ax=ax[1])

ax[0].set_title('SMOKE vs AGE')

ax[1].set_title('GENDER vs HEIGHT')

plt.show()
sns.swarmplot(y='LungCap(cc)',data=data,x='Caesarean')

plt.suptitle('LUNG CAPACITY vs CAESAREAN')

plt.show()
sns.jointplot(x='Age( years)',y='Height(inches)',data=data,kind='reg')

plt.suptitle('AGE vs HEIGHT')

plt.show()
fig,axes=plt.subplots(1,2,figsize=(20,6))

sns.scatterplot(y='Age( years)',data=data,x='LungCap(cc)',ax=axes[0])

sns.scatterplot(x='LungCap(cc)',data=data,y='Height(inches)',ax=axes[1])

axes[0].set_title('LUNG CAPACITY vs AGE')

axes[1].set_title('LUNG CAPACITY vs HEIGHT')

plt.show()
fig=plt.figure(figsize=(8,6))

sns.countplot(x='Gender',data=data,hue='Smoke')

plt.title('Gender vs Smoke')

plt.show()
fig=plt.figure(figsize=(10,6))

ct=pd.crosstab(index=data['Gender'],columns=data['Caesarean'])

plt.bar(ct.index,ct['yes'],label='Born by Caesarean')

plt.bar(ct.index,ct['no'],bottom=ct['yes'],label='Not Born by Caesarean')

plt.xlabel('Gender')

plt.title('GENDER vs CAESAREAN',color='purple')

plt.legend()

plt.show()
fig=plt.figure(figsize=(10,6))

ct=pd.crosstab(index=data['Caesarean'],columns=data['Smoke'])

plt.bar(ct.index,ct['yes'],label='Smokers')

plt.bar(ct.index,ct['no'],bottom=ct['yes'],label='Non Smokers')

plt.xlabel('Caesarean')

plt.title('SMOKE vs CAESAREAN',color='darkblue')

plt.legend()

plt.show()
fig,axes=plt.subplots(2,2,figsize=(20,15))

sns.violinplot(x='Age( years)',data=data,y='Smoke',hue='Caesarean',ax=axes[0,0])

axes[0,0].set_title('1. AGE vs SMOKE')

sns.violinplot(x='Gender',data=data,y='Age( years)',hue='Caesarean',ax=axes[0,1])

axes[0,1].set_title('2. GENDER vs AGE')

sns.swarmplot(y='LungCap(cc)',data=data,x='Age( years)',hue='Smoke',ax=axes[1,0])

axes[1,0].set_title('3. AGE vs LUNG CAPACITY')

sns.scatterplot(x='Height(inches)',data=data,y='LungCap(cc)',hue='Smoke',ax=axes[1,1])

axes[1,1].set_title('4. HEIGHT vs LUNGCAPACITY')

plt.show()
sns.lmplot(x='Age( years)',y='LungCap(cc)',data=data,hue='Gender',fit_reg=False,row='Smoke',col='Caesarean')

plt.show()
fig=plt.figure(figsize=(8,6))

sns.heatmap(data.corr(),cmap='Blues')

plt.title('HEAT MAP',color='darkblue')

plt.show()
sns.pairplot(data)

plt.show()
sns.pairplot(data,hue='Smoke',markers=['o','x']) # for histogram use diag_kind='hist' 

plt.show()
sns.pairplot(data,hue='Gender',palette='seismic',markers=['o','s']) # for kde use diag_kind='kde' 

plt.show()
data.mean()
data.median()
data.mode()
data.min()
data.max()
print(data[['Age( years)','LungCap(cc)','Height(inches)']].max()-data[['Age( years)','LungCap(cc)','Height(inches)']].min())
print('Q1:',data.quantile(q=0.25))
print('Q2:',data.quantile(q=0.5))
print('Q3:',data.quantile(q=0.75))
print('IQR:',data.quantile(0.75)-data.quantile(0.25))
data.var()
data.std()
data.mad()
data.describe()
skewness=data.skew()

skewness
kurtosis=data.kurt()

kurtosis
fig,axes=plt.subplots(1,2,figsize=(12,6))

skewness=data.skew()

s=pd.DataFrame(skewness,columns=['Skewness'])

stats.probplot(s['Skewness'].values,plot=plt)

sns.kdeplot(s['Skewness'].values,ax=axes[0],label='Skewness')

axes[0].set_title('KDE Plot')

kurtosis=data.kurt()

k=pd.DataFrame(kurtosis,columns=['Kurtosis'])

stats.probplot(k['Kurtosis'].values,plot=plt)

sns.kdeplot(k['Kurtosis'],ax=axes[1])

plt.show()