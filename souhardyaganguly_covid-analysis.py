# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing Dataset

data = pd.read_csv('/kaggle/input/covid19-corona-virus-india-dataset/tests_state_wise.csv')

data
data = data.fillna(0)

data.isna().any()


data.drop(['Updated On','Source1', 'Unnamed: 22', 'Source2','Tag (People in Quarantine)'

                       ,'Tag (Total Tested)', ], axis = 1, inplace = True)
data['Negative'] = data['Negative'].replace(' ', 0)

type(data['Negative'][0])
for i in range(0, len(data['Negative'])):

    data['Negative'][i] = int(data['Negative'][i])
type(data['Negative'][0])
#Checking 

for i in range(0, len(data['Negative'])):

    if type(data['Negative'][i]) != int:

        print(data['Negative'][i])
df = data.groupby('State').sum()

df.reset_index(inplace=True)

df.sort_values('Total Tested',inplace=True,ascending=False)

df.head()
df['Negative'] = df['Total Tested'] - (data['Positive']+ data['Unconfirmed'])

df.columns
type(df['Population NCP 2019 Projection'][0])
plt.figure(figsize = (10,10))

plt.bar(df['State'], df['Total Tested'])

plt.xticks(rotation = 90)

plt.xlabel('States', fontsize = 20)

plt.ylabel('Total Tested (Scale: 1:10^7)', fontsize = 20)

plt.show()
df['positive_per_thousand_tested'] = (df['Positive']/df['Total Tested'])*1000

plt.figure(figsize = (10,10))

plt.bar(df['State'], df['positive_per_thousand_tested'])

plt.xticks(rotation = 90)

plt.xlabel('States', fontsize = 20)

plt.ylabel('Positive per thousand tested', fontsize = 20)

plt.show()
plt.figure(figsize = (10,10))

plt.bar(df['State'], df['Negative'])

plt.xticks(rotation = 90)

plt.xlabel('States', fontsize = 20)

plt.ylabel('Negative', fontsize = 20)

plt.show()
plt.figure(figsize = (10,10))

plt.bar(df['State'], df['Cumulative People In Quarantine'])

plt.xticks(rotation = 90)

plt.xlabel('States', fontsize = 20)

plt.ylabel('Cumulative People In Quarantine', fontsize = 20)

plt.show()
plt.figure(figsize=(16,9))

#set bar height

bar_width=.40

#set postion on x axis

r1=np.arange(len(df['Total Tested']))

r2=[x +bar_width for x in r1]



#ploting the bar graph

plt.bar(r1,df['Total Tested'],color='#7f6d5f',width=bar_width,edgecolor='white',label='Total tested')

plt.bar(r2,df['Positive'],color='#557f2d',width=bar_width,edgecolor='white',label='Total positive')



# adding the xticks

plt.xlabel('States',fontweight='bold')

plt.xticks([r+bar_width for r in range(len(df['Total Tested']))],df['State'],rotation='vertical')

plt.ylabel('Scale Log')

plt.yscale('log')

plt.legend()

plt.title('Total Positive and Total Tested',)

plt.show()
plt.figure(figsize=(16,9))

#set bar height

bar_width=.40

#set postion on x axis

r1=np.arange(len(df['Total Tested']))

r2=[x +bar_width for x in r1]



#ploting the bar graph

plt.bar(r1,((df['Total Tested']/df['Population NCP 2019 Projection'])*1000),color='#7f6d5f',width=bar_width,edgecolor='white',label='Total tested per thousand')

plt.bar(r2,(df['Positive']/df['Population NCP 2019 Projection']*1000),color='#557f2d',width=bar_width,edgecolor='white',label='Total positive per thousand')



# adding the xticks

plt.xlabel('States',fontweight='bold', fontsize = 20)

plt.xticks([r+bar_width for r in range(len(df['Total Tested']))],df['State'],rotation='vertical')

plt.ylabel('Scale Log', fontsize = 20)

plt.yscale('log')

plt.legend()

plt.title('Total Positive and Total Tested',)

plt.show()