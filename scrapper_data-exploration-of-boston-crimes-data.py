import pandas as pd

import numpy as np



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

data = pd.read_csv('/kaggle/input/crimes-in-boston/crime.csv', encoding = 'latin-1')

data.head()
data = pd.read_csv('/kaggle/input/crimes-in-boston/crime.csv', encoding = 'latin-1')

data.head()
data.shape
data.describe()
data.dtypes
data.columns
x = data.drop(['OFFENSE_CODE','OFFENSE_DESCRIPTION','OCCURRED_ON_DATE','Lat','Long','Location'],axis=1)

x.shape
x.rename(str.lower, axis =1, inplace = True)

x.head()
x.district.unique()
import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



plt.figure(figsize = (10,6),dpi=80, facecolor='w', edgecolor='k')

plt.title(' crimes in years')

#ax1 = plt.subplot(1,2,1)

sns.countplot('year',data = x )
plt.figure(figsize = (10,6))

plt.title('Total crimes in various months')

sns.countplot(x = 'month', data =x)

plt.tight_layout()
plt.figure(figsize = (10,6))

plt.title('Total crimes across different days of week')

sns.countplot(x = 'day_of_week', data =x)
plt.figure(figsize = (10,6))

plt.title(' crimes at different hours')

#plt.ylim(0,12)

sns.countplot(x = 'hour', data =x, palette='rainbow')
plt.figure(figsize = (10,6))

plt.title('crime variation (ucr_part) in months')

plt.ylim(0,18000)

sns.countplot(x = 'month', data =x, hue = 'ucr_part', palette='rainbow')
plt.figure(figsize = (10,6))

plt.title('crimes variation  in months for diff years')

plt.ylim(0,10000)

sns.countplot(x = 'month', data =x, hue = 'year', palette='rainbow')
plt.figure(figsize = (10,6))

plt.title(' crimes(ucr_part) in different days')

#plt.ylim(0,18000)

sns.countplot(x = 'day_of_week', data =x, palette='rainbow',hue ='ucr_part')
order = x.ucr_part.unique()[:3]
plt.figure(figsize = (10,6))

plt.title(' crimes in different days for diff years')

#plt.ylim(0,18000)

sns.countplot(x = 'day_of_week', data =x, palette='rainbow',hue ='year')
plt.figure(figsize = (10,6))

sns.countplot('ucr_part',data = x, order = order, hue='year')
x.day_of_week.value_counts()
plt.figure(figsize = (12,6))

plt.subplot(1,2,1)

sns.countplot(x='district',data = x, hue ='year',palette= 'Set1')

plt.subplot(1,2,2)

sns.countplot(x='district',data = x, hue ='ucr_part', palette= 'rainbow')



plt.tight_layout()
x['ucr_part'] = x.ucr_part.astype('category')

x['year']  = x.year.astype('category')
tot = x.groupby(['offense_code_group','year']).year.count()

type(tot)
crime = tot.unstack()

crime.sort_values(by=[2015,2016,2017,2018], ascending= False).head()
nme = crime.sort_values(by=[2015,2016,2017,2018], ascending= False).head().index
sns.catplot(x='hour',col='offense_code_group',kind="count",col_wrap=3,

              data= x[x['offense_code_group'].isin(nme)])