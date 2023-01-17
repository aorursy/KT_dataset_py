# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(style='dark')
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('../input/spotify.csv', encoding='ISO-8859-1')

data.head()

#Dropping the Unnamed: 0 column
data.drop('Unnamed: 0', axis=1, inplace=True)
data.info()
#Checking if there are any null/missing values in the dataset
data.isna().sum()
sns.heatmap(data.isnull(), cbar=False)
#Changing a few column names for better readability
data.rename(columns={'title':'song', 'artist':'artist', 'top genre':'genre', 'year':'year', 'bpm':'beats_per_minute','nrgy':'energy',
                    'dnce':'danceability','dB':'loudness','live':'liveness', 'val':'valence', 'dur':'length', 'acous':'acousticness',
                    'spch':'speechiness','pop':'popularity'}, inplace=True)
data.columns
data.describe()
data['genre'].nunique()
data['genre'].value_counts().head(10)
data['genre'].value_counts().head(10).plot.pie(figsize=(10,10))
data['artist'].nunique()
data['artist'].value_counts().head(10)
data['artist'].value_counts().head(10).plot.bar()
data.beats_per_minute.describe()
def grouping(x):
    if x<=100:
        return '<100'
    elif x<=150:
        return '101-150'
    elif x<=200:
        return '151-200'
    else:
        return '>200'
    
groups = data.beats_per_minute.apply(grouping)
values = groups.value_counts()
labels = values.index
fig = px.pie(data, values=values,names=labels)
fig.update_layout(title = 'BPM Distribution')
fig.show()
data[data['beats_per_minute']>200]
fig= px.scatter(data[data['beats_per_minute']>200], x='popularity',
y='beats_per_minute', hover_name='song', color='beats_per_minute',
               size='acousticness')
fig.show()
fig = px.violin(data, y="danceability", color="year", points='all', hover_name='song', hover_data=['artist'])
fig.show()
fig = px.violin(data, y="danceability", color="year", points='all', hover_name='song', hover_data=['artist'])
fig.show()
fig = px.violin(data, y="energy", color="year", points='all', hover_name='song', hover_data=['artist'])
fig.show()

fig = px.scatter(data, x='danceability', y='energy',color='energy',
                hover_name='song',hover_data=['artist','year'])
fig.show()
fig = px.scatter(data, x="popularity", y="length", color='length', hover_name='song', hover_data=['artist','year'])
fig.show()
fig = px.scatter(data, x="popularity", y="speechiness", color='year', hover_name='song', hover_data=['artist','year','length'])
fig.show()
fig = px.scatter(data.query('year==2019'),x='artist',y='popularity',
                 hover_name='song',color='popularity')
fig.show()
fig = px.scatter(data[data['genre'].str.contains('hip hop')], x="artist", y="popularity", hover_name='song')
fig.show()
kp = data[data['artist']=='Katy Perry']
kp
kp.song.count()
data[data['artist']=='Katy Perry']['year'].value_counts()
fig = px.scatter(data[data['artist']=='Katy Perry'], x='popularity',y='year',
                hover_name='song',hover_data=['artist','year'])
fig.show()
data.sort_values(by='popularity',ascending=False).head(10)[['song','artist','year','popularity']]
correlations = data.corr()
fig = plt.figure(figsize=(14,10))
sns.heatmap(correlations, annot=True, cmap='GnBu_r',center=1)
le = LabelEncoder()
for column in data.columns.values:
  if data[column].dtypes == 'object':
    le.fit(data[column].values)
    data[column] = le.transform(data[column])

data.head()

X = data.drop('loudness',axis=1)
y=data.loudness
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print(mean_squared_error(y_test, y_pred))
svr= SVR(C=0.5)
svr.fit(X_train, y_train)

y_pred=svr.predict(X_test)
print(mean_squared_error(y_test,y_pred))
