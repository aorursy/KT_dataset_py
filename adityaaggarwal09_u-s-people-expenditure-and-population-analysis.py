# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O(e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import missingno as mis

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/farmers-markets-in-the-united-states/wiki_county_info.csv')

info=pd.read_csv('/kaggle/input/farmers-markets-in-the-united-states/farmers_markets_from_usda.csv')
data.head()
info.head()
data.info()
data.describe()
sns.heatmap(data.isnull(),cmap='Blues')
data.isnull().sum()
data.dropna(inplace=True)
data['county'].value_counts()
plt.figure(figsize=(8,6))

sns.barplot(x=data['county'].value_counts()[0:30],y=data['county'].value_counts()[0:30].index)
data['State'].nunique()
plt.figure(figsize=(8,6))

sns.barplot(x=data['State'].value_counts()[0:30],y=data['State'].value_counts()[0:30].index,palette="ch:.25")
def remove_char(df):

    bad_var = ['per capita income', 'median household income', 'median family income', 'population', 'number of households']

    bad_tokens = ['$', r',']

    

    for var in bad_var:

        df[var] = df[var].replace('[\$,]', '', regex=True).astype(int)

    return df
data=remove_char(data)
plt.figure(figsize=(20,6))

plt.plot(data['population'])

plt.title('Variation of Population')
plt.figure(figsize=(20,6))

plt.plot(data['number of households'])

plt.title('Variation of No of Households')
plt.figure(figsize=(10,4))

axes=sns.lmplot(x='population',y='number of households',data=data,palette='coolwarm')
data['Avg_person']=data['population']/data['number of households']

avg=data['Avg_person'].sum()/len(data)

print(avg)
data['Avg_person']=data['Avg_person'].round(0).astype(int)
sns.countplot(x=data['Avg_person'])
plt.figure(figsize=(6,13))

sns.stripplot(x='Avg_person',y='State',data=data,jitter=True) 
plt.figure(figsize=(20,6))

plt.title('Variation of Per Capita Income')

plt.plot(data['per capita income'])
plt.figure(figsize=(20,6))

plt.title('Variation of Median Household Income')

plt.plot(data['median household income'])
plt.figure(figsize=(20,6))

plt.title('Variation of Median Family Income')

plt.plot(data['median family income'])
plt.figure(figsize=(20,8))

sns.lineplot(data=data.iloc[:,[3,4,5]])
county=data.groupby(['county']).mean()

state=data.groupby(['State']).mean()
county.head()
state.head()
state_PCI=state.sort_values(by=['per capita income'],ascending=False)

state_PCI=state_PCI.head(10)

state_PCI=state_PCI.reset_index()

plt.figure(figsize=(16,5))

sns.barplot(x='State',y='per capita income',data=state_PCI)

plt.title('Top 10 states having higest Per Capita Income')
state_MHI=state.sort_values(by=['median household income'],ascending=False)

state_MHI=state_MHI.head(10)

state_MHI=state_MHI.reset_index()

plt.figure(figsize=(16,5))

sns.barplot(y='State',x='median household income',data=state_MHI)

plt.title('Top 10 states having higest Median Household Income')
state_MFI=state.sort_values(by=['median family income'],ascending=False)

state_MFI=state_MFI.head(10)

state_MFI=state_MFI.reset_index()

plt.figure(figsize=(16,5))

sns.barplot(x='State',y='median family income',data=state_MFI)

plt.title('Top 10 states having higest Median Family Income')
state_pop=state.sort_values(by=['population'],ascending=False)

state_pop=state_pop.head(10)

state_pop=state_pop.reset_index()

plt.figure(figsize=(16,5))

sns.barplot(x='population',y='State',data=state_pop)

plt.title('Top 10 states having higest Population')
county_PCI=county.sort_values(by=['per capita income'],ascending=False)

county_PCI=county_PCI.head(10)

county_PCI=county_PCI.reset_index()

plt.figure(figsize=(22,5))

sns.scatterplot(x='county',y='per capita income',data=county_PCI)

plt.title('Top 10 county having higest Per Capita Income')
county_MHI=county.sort_values(by=['median household income'],ascending=False)

county_MHI=county_MHI.head(10)

county_MHI=county_MHI.reset_index()

plt.figure(figsize=(22,5))

sns.barplot(x='county',y='median household income',data=county_MHI)

plt.title('Top 10 county having higest Median Household Income')
county_MFI=county.sort_values(by=['median family income'],ascending=False)

county_MFI=county_MFI.head(10)

county_MFI=county_MFI.reset_index()

plt.figure(figsize=(22,5))

sns.barplot(x='county',y='median family income',data=county_MFI)

plt.title('Top 10 county having higest Median Family Income')
county_pop=county.sort_values(by=['population'],ascending=False)

county_pop=county_pop.head(10)

county_pop=county_pop.reset_index()

plt.figure(figsize=(22,5))

sns.barplot(x='population',y='county',data=county_pop)

plt.title('Top 10 county having higest Median Family Income')
sns.pairplot(data,palette='coolwarm')
plt.figure(figsize=(15,6))

sns.heatmap(data.corr(),annot=True,cmap='coolwarm')
plt.figure(figsize=(24,8))

sns.regplot(x='per capita income',y='median household income',data=data,color='orange')

plt.title('Per Capita Income vs Median Household Income')
plt.figure(figsize=(24,8))

sns.regplot(x='per capita income',y='median family income',data=data,color='blue')

plt.title('Per Capita Income vs Median Family Income')
plt.figure(figsize=(24,8))

sns.regplot(x='number of households',y='population',data=data,color='green')

plt.title('Population vs No of Households')