# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from matplotlib import rcParams
import warnings 
warnings.filterwarnings("ignore")
%matplotlib inline
plt.style.use('fivethirtyeight')
#plt.style.use('bmh')
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['text.color'] = 'k'
playstore = pd.read_csv("../input/googleplaystore.csv")
user_reviews = pd.read_csv("../input/googleplaystore_user_reviews.csv")
playstore.head()
playstore.drop_duplicates(subset = ['App'],keep='first',inplace=True)
playstore.reset_index(inplace=True)
playstore.info()
playstore['Price'].unique()
playstore = playstore[playstore['Price']!= 'Everyone'].reset_index()
for i in range(0,len(playstore['Price'])):
    if '$' in playstore.loc[i,'Price']:
        playstore.loc[i,'Price'] = playstore.loc[i,'Price'][1:]
    playstore.loc[i,'Price'] =  float(playstore.loc[i,'Price'])
playstore['Price'].unique()
playstore.head()
playstore['Reviews'] = playstore['Reviews'].astype('float')
playstore['Price'] = playstore['Price'].astype('float')
playstore.info()
playstore.head()
playstore["Rating"] = playstore.groupby("Category")['Rating'].transform(lambda x: x.fillna(x.mean()))
playstore.head()
f,ax1 = plt.subplots(ncols=1)
sns.countplot("Category", data=playstore,ax=ax1,order=playstore['Category'].value_counts().index)
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
f.set_size_inches(25,10)
ax1.set_title("Count by Categories")
f,(ax1,ax2,ax3) = plt.subplots(ncols=3,sharey=False)
sns.distplot(playstore['Rating'],hist=True,color='b',ax=ax1)
sns.distplot(playstore['Reviews'],hist=True,color='r',ax=ax2)
sns.distplot(playstore['Price'],hist=True,color='g',ax=ax3)
f.set_size_inches(15, 5)
f,(ax1,ax2,ax3) = plt.subplots(ncols=3,sharey=False)
sns.boxplot(x='Rating',data=playstore,ax=ax1)
sns.boxplot(x='Reviews',data=playstore,ax=ax2)
sns.boxplot(x='Price',data=playstore,ax=ax3)
f.set_size_inches(15, 5)
sns.countplot(x='Type',data=playstore)
f,ax1 = plt.subplots(ncols=1)
sns.countplot(x = 'Installs',hue='Content Rating',data=playstore,ax=ax1)
plt.xticks(rotation=90)
f.set_size_inches(15,5)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

f,ax1 = plt.subplots(ncols=1)
sns.countplot(x = 'Installs',hue='Type',data=playstore,ax=ax1,order=playstore['Installs'].value_counts().index)
plt.xticks(rotation=90)
f.set_size_inches(15,5)
f,ax1 = plt.subplots(ncols=1)
sns.boxplot(x = 'Category',y='Rating',data=playstore)
plt.xticks(rotation=90)
f.set_size_inches(15,5)
f,ax1 = plt.subplots(ncols=1)
sns.boxplot(x = 'Installs',y='Rating',data=playstore)
plt.xticks(rotation=90)
f.set_size_inches(15,5)
playstore.head()
g = sns.lmplot(x = 'Reviews',y='Rating',data=playstore)
playstore['Size'].replace('Varies with device','0',inplace=True)
for i in range(0,len(playstore['Size'])):
    if 'k' in playstore.loc[i,'Size']:
        playstore.loc[i,'Size'] = playstore.loc[i,'Size'][:-1]
        playstore.loc[i,'Size'] = float(playstore.loc[i,'Size']) / 1000
    elif 'M' in playstore.loc[i,'Size'] :
        playstore.loc[i,'Size'] = playstore.loc[i,'Size'][:-1]
        playstore.loc[i,'Size'] = float(playstore.loc[i,'Size'])
playstore['Size'] = playstore['Size'].astype('float')
playstore['Size'].replace(0,np.nan,inplace=True)
playstore['Size'] = playstore.groupby(by = 'Category')['Size'].transform(lambda x: x.fillna(x.mean()))
for i in range(0,len(playstore['Installs'])):
    if '+' in playstore.loc[i,'Installs']:
        playstore.loc[i,'Installs'] = playstore.loc[i,'Installs'][:-1]
        playstore.loc[i,'Installs'] = playstore.loc[i,'Installs'].replace(',','')
        playstore.loc[i,'Installs'] = float(playstore.loc[i,'Installs'])
playstore['Installs'] = playstore['Installs'].astype('float')
g = sns.scatterplot(y='Size',x='Installs',hue = 'Category',data=playstore[playstore['Category'].isin(['FAMILY','GAMES','TOOLS','BUSINESS','MEDICAL'])])
g = sns.catplot(x = 'Category',y='Size',kind='boxen',data=playstore,height=5,aspect=2)
g.set_xticklabels(rotation=90)
g = sns.catplot(x = 'Content Rating',y='Installs',kind='boxen',data=playstore,height=5,aspect=2)
g.set_xticklabels(rotation=90)
playstore.head()
playstore.groupby('Category')['Rating'].mean().sort_values(ascending=False)
