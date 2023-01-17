# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

%matplotlib inline 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/top2018.csv')
df.tail()
df.columns
df['Duration_min']=df['duration_ms']/60000

df.drop(columns='duration_ms',inplace=True)
# added a popularity column.

df['popularity'] = df.index + 1

df.head()
sns.pairplot(df, x_vars='popularity' , y_vars=['danceability', 'energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','Duration_min','time_signature']);
sns.heatmap(df.corr(),cmap="YlOrRd")
Correlation=df[['danceability','energy','valence','loudness','tempo']]

sns.heatmap(Correlation.corr(),annot=True,cmap="YlOrRd")
df['artists'].value_counts().head(10)
df['TF']=df['popularity']<=12 #top 50

df.columns

top50 = df[['danceability', 'energy', 'key', 'loudness',

       'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',

       'valence', 'tempo', 'time_signature', 'Duration_min', 'popularity',

       'TF']]

top50.head()
top50.tail()
y = top50[['TF']]

x = top50[['danceability', 'energy', 'key', 'loudness',

       'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',

       'valence', 'tempo', 'time_signature', 'Duration_min']]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

x_train.shape, x_test.shape, y_train.shape, y_test.shape
from sklearn.ensemble import RandomForestClassifier as rf

clf = rf(n_estimators=100, max_depth=2,random_state=0)

clf.fit(x_train, y_train)
d = list(top50.columns)

xx = list(clf.feature_importances_)

for i in range(13):

    print(d[i], xx[i], sep=' :: ')

clf.score(x_train, y_train)
clf.score(x_test, y_test)