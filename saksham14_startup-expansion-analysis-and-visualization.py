# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_excel('/kaggle/input/P1-StartupExpansion.xlsx')
data.head()
data.shape
data.describe()
data.info()
data.drop('Store ID',axis=1,inplace=True)
data.tail(5)
sns.heatmap(data.isnull(),cbar=False,yticklabels=False,cmap='cividis')
plt.style.use('ggplot')

sns.boxplot(data['Sales Region'],data['Revenue'],hue=data['New Expansion'])
data['New Expansion'].value_counts()
data.groupby('State').describe()['Revenue']['mean'].round(0)
# plt.rcParams['font.size'] = 9.0

plt.style.use('dark_background')

fig = plt.figure(figsize=(15,15))

sns.barplot(y=data['State'].unique(),x=data.groupby('State').describe()['Revenue']['mean'].round(0),data=data)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.xlabel('States',fontSize=20)

plt.ylabel('Average Revenue',fontSize=20)

plt.title('Average Revenue per State',fontSize=32)

plt.tight_layout()

plt.savefig('Average Revenue.png')
sns.set()

sns.scatterplot(x='Revenue',y='Marketing Spend',data=data,hue='Sales Region')

plt.title('Comparison b/w Revenue and Marketing Spend')

# plt.legend(loc=(1.05,0.75))
sns.heatmap(data.corr(),annot=True)
sns.set(style='darkgrid')

sns.swarmplot(x='Sales Region',y='Marketing Spend',data=data)
#City with Highest Revenue

data[data['Revenue']==data['Revenue'].max()][['City','State','Marketing Spend','Revenue']]
#City with Lowest Revenue

data[data['Revenue']==data['Revenue'].min()][['City','State','Marketing Spend','Revenue']]
#City with Highest Marketing Spend

data[data['Marketing Spend']==data['Marketing Spend'].max()][['City','State','Marketing Spend','Revenue']]
#City with Lowest Marketing Spend

data[data['Marketing Spend']==data['Marketing Spend'].min()][['City','State','Marketing Spend','Revenue']]
#Cities with low marketing spend than average and high revenue than average

major_data = data[(data['Marketing Spend']<data['Marketing Spend'].mean()) & (data['Revenue']>data['Revenue'].mean())]
bar_data = major_data.groupby('State',as_index=False).count().sort_values(by='Revenue',ascending=False).reset_index()[['State','City']]



# By this data we can say that California is most suitable city for most successfull startups with low spend,followed by Washington
sns.scatterplot(x='Revenue',y='Marketing Spend',data=major_data,hue='Sales Region')

plt.xlim(data['Revenue'].min(),data['Revenue'].max())

plt.ylim(data['Marketing Spend'].min(),data['Marketing Spend'].max())

plt.title('States with low Marketing Bugdet and High Profit Based on Region')
plt.figure(figsize=(10,8))

plt.style.use('fivethirtyeight')

sns.barplot(x='City',y='State',data=bar_data,color='blue')

plt.title('Most Successful States in terms of less spend and high profit')

plt.xlabel('Count')