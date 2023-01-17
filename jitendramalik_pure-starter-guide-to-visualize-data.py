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
import pandas as pd

import numpy as np

import csv

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder



filename='/kaggle/input/top50spotify2019/top50.csv'

data=pd.read_csv(filename,encoding='ISO-8859-1')



#print(data.head())

#print(data.describe())

#print(data.columns)

#printing null values



#print(((data.isnull().sum())/data.shape[0])*100)



sns.pairplot(data)

sns.set(font_scale=1.15)

cor=data.corr()

plt.tight_layout()

plt.show()

#le=LabelEncoder()

#plt.figure(figsize=(16,6))

data["Track.Name"]=data['Track.Name'].str.split("(",n=-1,expand=True)[0]



#------------------------plotting most hit genre

ax=sns.countplot(x='Genre',data=data,order = data['Genre'].value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha="right",fontsize=8)

print(data['Artist.Name'].unique())

print(data['Genre'].unique())

plt.show()





#-------------------------plotting most hit artist

ax=sns.countplot(x='Artist.Name',data=data,order=data['Artist.Name'].value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha="right",fontsize=8)

plt.tight_layout()

plt.show()





#-------------------------popularity and the track name

ax=sns.barplot(x='Popularity',y='Track.Name',data=data)

#ax.sns.load_dataset('Popularity').sort_values(ascending=True)

sns.despine(left=True,bottom=True)

plt.tight_layout()

plt.show()





datacor=data.drop((['Unnamed: 0']),axis=1)



#------------------------coorelation of popularity with all the variables

datacor.corr().loc['Popularity',:]

sns.heatmap(datacor.corr().loc[['Popularity'],:],annot=True)

#print(corr)

plt.show()





#-----------------------corrlation of all variable

sns.heatmap(datacor.corr(),annot=True)

plt.show()