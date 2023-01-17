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
import matplotlib.pyplot as plt

import seaborn as sns

data_15 = pd.read_csv('/kaggle/input/world-happiness/2015.csv')

data_16 = pd.read_csv('/kaggle/input/world-happiness/2016.csv')

data_17 = pd.read_csv('/kaggle/input/world-happiness/2015.csv')
data_15.head(10)
data_15.info()
data_15.describe()
data_16.describe()
data_17.describe()
data_15.drop(['Standard Error'],axis=1,inplace=True)

data_17.drop(['Standard Error'],axis=1,inplace=True)

data_16.drop(['Lower Confidence Interval','Upper Confidence Interval'],axis=1,inplace=True)
data_15.describe()

data_16.describe()
data_17.describe()
data = pd.concat([data_15,data_16,data_17])

data.rename(columns={'Economy (GDP per Capita)':'GDP/Capita'},inplace=True)
data.describe()
data.head()
data['Country'].nunique()
data.groupby('Region').mean().sort_values(by='Happiness Score',ascending=False)
data.groupby('Region').mean().sort_values(by='Happiness Rank')
plt.figure(figsize=(25,5))

sns.countplot('Region',data=data )
data[['Region','Happiness Rank']].groupby('Region').aggregate(['count', np.mean]).sort_values(by=[('Happiness Rank','mean')])
data[['Region','Happiness Rank']].groupby('Region').aggregate(['count', np.mean]).corr()
data[data['Country']=='Vietnam']
sns.jointplot('Generosity','Happiness Rank',data=data, kind = 'reg')
sns.set_style('whitegrid')

sns.distplot(data['GDP/Capita'])
data.set_index(['Region','Country'])
data[['Region','GDP/Capita','Happiness Rank']].groupby(by='Region',as_index=False).mean().sort_values(by='GDP/Capita',ascending=False)
data[['GDP/Capita','Happiness Rank']].corr()
sns.jointplot('GDP/Capita','Happiness Rank',data=data, kind = 'reg')
sns.heatmap(data.corr(),annot=True)
sns.heatmap(data[data['Region']=='Australia and New Zealand'].corr(),annot=True)
sns.heatmap(data[data['Region']=='North America'].corr(),annot=True)
sns.heatmap(data[data['Region']=='Western Europe'].corr(),annot=True)
sns.heatmap(data[data['Region']=='Southeastern Asia'].corr(),annot=True)
sns.heatmap(data[data['Region']=='Middle East and Northern Africa'].corr(),annot=True)
sns.heatmap(data[data['Region']=='Latin America and Caribbean'].corr(),annot=True)
df = data_15

df_1=pd.merge(df,data_16,on='Country')

df_1 = df_1[['Country','Happiness Rank_x','Happiness Score_x','Happiness Rank_y', 'Happiness Score_y']]



df_1
df_1['Difference in rank'] = df_1['Happiness Rank_x'].sub(df_1['Happiness Rank_y'])

df_1['Difference in score'] = df_1['Happiness Score_x'].sub(df_1['Happiness Score_y'])

df_1
sns.distplot(df_1['Difference in rank'])
sns.distplot(df_1['Difference in score'])
df_1['Difference in rank'].describe()
df_1['Difference in score'].describe()
df_1[df_1['Difference in rank']>10][['Country','Happiness Rank_x','Happiness Rank_y','Difference in rank']].sort_values(by='Difference in rank',ascending=False)
df_1[df_1['Difference in rank']<-10][['Country','Happiness Rank_x','Happiness Rank_y','Difference in rank']].sort_values(by='Difference in rank')