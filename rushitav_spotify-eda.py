# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/top50spotify2019/top50.csv',encoding='ISO-8859-1')

df=data.sort_values(by='Popularity',ascending=False)

df
df.drop('Unnamed: 0',axis=1,inplace=True)

df.head()
df.isnull().sum()
df.describe()
df.shape

df.drop('Track.Name', axis=1, inplace=True)



df.columns=['Artist_Name','Genre','Beats_Per_Minute','Energy','Danceability','Loudness','Liveness','Valence','Length','Acoustiness','Speechiness','Popularity']
df.Genre.value_counts()
fig=plt.figure(figsize=(25,5))

sns.countplot('Genre',data=df)

x=plt.gca().xaxis

for item in x.get_ticklabels():

    item.set_rotation(45)

    plt.subplots_adjust(bottom=0.25)

plt.show()
df.Artist_Name.value_counts()
fig=plt.figure(figsize=(25,5))

sns.countplot('Artist_Name',data=df)

x=plt.gca().xaxis

for item in x.get_ticklabels():

    item.set_rotation(90)

    plt.subplots_adjust(bottom=0.25)

plt.show()
df.head()
corr=df.corr()

plt.figure(figsize=(15,10))

sns.heatmap(corr,annot=True)
fig=sns.pairplot(df)

fig.map_lower(sns.regplot)

plt.show()