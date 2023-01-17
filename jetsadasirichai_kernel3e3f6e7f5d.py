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
import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
z = pd.read_csv('../input/top-spotify-tracks-of-2018/top2018.csv')

z.head()
z.info()
sns.heatmap(z.corr(),cmap="YlOrRd")
z['artists'].value_counts().head(10)
sns.set_style(style='darkgrid')

sns.distplot(z['danceability'],hist=True,kde=True)
sns.distplot(z['energy'])

Ve=z['energy']>=0.75

Re=(z['energy']>=0.5) & (z['energy']<0.75)

Le=z['energy']<0.5
data=[Ve.sum(),Re.sum(),Le.sum()]

Energy=pd.DataFrame(data,columns=['percent'],

                   index=['Very Energy','Regular Energy','Low Energy'])
Energy
Correlation=z[['danceability','energy','valence','loudness','tempo']]
sns.heatmap(Correlation.corr(),annot=True,cmap="YlOrRd")
Mayores=z[z['mode']==1]

Menores=z[z['mode']==0]
MayoresD=Mayores[Mayores['danceability']>=0.5]

MenoresD=Menores[Menores['danceability']>=0.5]
MayoresD=Mayores.drop(columns=['mode','time_signature'])

MenoresD=Menores.drop(columns=['mode','time_signature'])
sns.heatmap(MayoresD.corr(),cmap="YlOrRd")
sns.heatmap(MenoresD.corr(),cmap="YlOrRd")
MaycorD=MayoresD[['danceability','energy','valence','loudness','tempo']]

MencorD=MenoresD[['danceability','energy','valence','loudness','tempo']]
sns.heatmap(MaycorD.corr(),annot=True,cmap="YlOrRd")
sns.heatmap(MencorD.corr(),annot=True,cmap="YlOrRd")
z.loc[ z['key']==0 ,'key']='C'    

z.loc[ z['key']==1 ,'key']='C#'    

z.loc[ z['key']==2 ,'key']='D'    

z.loc[ z['key']==3 ,'key']='D#'    

z.loc[ z['key']==4 ,'key']='E'    

z.loc[ z['key']==5 ,'key']='F'    

z.loc[ z['key']==6 ,'key']='F#'    

z.loc[ z['key']==7 ,'key']='G'    

z.loc[ z['key']==8 ,'key']='G#'    

z.loc[ z['key']==9 ,'key']='A'    

z.loc[ z['key']==10 ,'key']='A#' 

z.loc[ z['key']==11 ,'key']='B' 
sns.set_style(style='darkgrid')

keys=z['key'].value_counts()

key_DF=pd.DataFrame(keys)

sns.barplot(x=key_DF.key, y=key_DF.index, palette="viridis")

plt.title('Popular keys')
z[['danceability','energy','valence','key']].groupby(by='key').mean().sort_values(by='danceability',ascending=False)