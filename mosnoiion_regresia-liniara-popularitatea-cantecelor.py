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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
data = pd.read_csv('/kaggle/input/girlsgoit-competition-2020/train.csv')
test = pd.read_csv('/kaggle/input/girlsgoit-competition-2020/test.csv')
submit = pd.read_csv('/kaggle/input/girlsgoit-competition-2020/sampleSubmission.csv')
data.head(20)
data.describe()
data.info()
# data.corr()
plt.figure(figsize=(18,18))
sns.heatmap(data.corr(), cmap="YlGnBu", annot = True)
plt.show()
X, y = data[['year']], data[['popularity']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)
reg1 = LinearRegression().fit(X_train, y_train)
print("Score on train:")
print("R^2", reg1.score(X_train, y_train))
print("MAE:", mean_absolute_error(y_train, reg1.predict(X_train)))

print()

print("Score on test:")
print("R^2", reg1.score(X_test,y_test))
print("MAE:", mean_absolute_error(y_test, reg1.predict(X_test)))
from sklearn.metrics import accuracy_score, f1_score, recall_score

Y_predict = reg1.predict(X_test).astype('int32')
print("Accuracy score:", accuracy_score(y_test, Y_predict))
print("F1 score:", f1_score(y_test, Y_predict, average='micro'))
print("Recall score:", recall_score(y_test, Y_predict, average='micro'))
data.columns
X, y = data[['acousticness', 'danceability', 'duration_ms',
       'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 'loudness',
       'mode', 'speechiness', 'tempo', 'valence',
       'year']], data[['popularity']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
reg1 = LinearRegression().fit(X_train, y_train)
print("Score on train:")
print("R^2", reg1.score(X_train, y_train))
print("MAE:", mean_absolute_error(y_train, reg1.predict(X_train)))

print()

print("Score on test:")
print("R^2", reg1.score(X_test,y_test))
print("MAE:", mean_absolute_error(y_test, reg1.predict(X_test)))
importance = reg1.coef_
features = ['acousticness', 'danceability', 'duration_ms',
       'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 'loudness',
       'mode', 'speechiness', 'tempo', 'valence',
       'year']
# summarize feature importance
for i,v in enumerate(importance[0]):
    print(features[i],v)
from matplotlib import pyplot
# plot feature importance
pyplot.bar([x for x in range(len(importance[0]))], importance[0])
pyplot.show()#0 - 1
Y_predict = reg1.predict(X_test).astype('int32')
print("Accuracy score:", accuracy_score(y_test, Y_predict))
print("F1 score:", f1_score(y_test, Y_predict, average='micro'))
print("Recall score:", recall_score(y_test, Y_predict, average='micro'))
from sklearn.preprocessing import Normalizer

transformer = Normalizer().fit(X_train, 'max')  # fit does nothing.


reg1 = LinearRegression().fit(transformer.transform(X_train), y_train)
print("Score on train:")
print("R^2", reg1.score(transformer.transform(X_train), y_train))
print("MAE:", mean_absolute_error(y_train, reg1.predict(transformer.transform(X_train))))

print()

print("Score on test:")
print("R^2", reg1.score(transformer.transform(X_test),y_test))
print("MAE:", mean_absolute_error(y_test, reg1.predict(transformer.transform(X_test))))
X1 = X_train.copy()
X1['year'] = (X1['year']-X1['year'].min())/(X1['year'].max()- X1['year'].min())
# X1['year']
def transform_custom(X1):
    for col in features:
        X1[col] = (X1[col]-X1[col].min())/(X1[col].max()- X1[col].min())
    return X1
X_train = transform_custom(X_train)
X_train.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
reg1 = LinearRegression().fit(transform_custom(X_train), y_train)
print("Score on train:")
print("R^2", reg1.score(transform_custom(X_train), y_train))
print("MAE:", mean_absolute_error(y_train, reg1.predict(transform_custom(X_train))))

print()

print("Score on test:")
print("R^2", reg1.score(transform_custom(X_test),y_test))
print("MAE:", mean_absolute_error(y_test, reg1.predict(transform_custom(X_test))))
importance = reg1.coef_
features = ['acousticness', 'danceability', 'duration_ms',
       'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 'loudness',
       'mode', 'speechiness', 'tempo', 'valence',
       'year']
# summarize feature importance
for i,v in enumerate(importance[0]):
    print(features[i],v)
# plot feature importance
pyplot.bar([x for x in range(len(importance[0]))], importance[0])
pyplot.show()#0 - 1
data.head(20)
max_per_year = data.groupby(['year'])['popularity'].apply(lambda x: x.value_counts().index[0]).reset_index()
max_per_year
plt.plot(max_per_year['year'],max_per_year['popularity'])
# hai sa mai facem careva date
import json
data['artists_array'] = data['artists'].apply(lambda x: [i.strip() for i in x.replace('[','').replace(']','').split(',')])#json.loads(x) if type(x) is str else []) 
data['artists_nr'] = data['artists_array'].apply(len)
data.head()
data[data['artists_nr']>1]#['artists_nr'].hist()
data['artists_nr'].hist()
df = data[['artists_array', 'popularity']]
ar = df['artists_array'].to_list()
score = df['popularity'].to_list()

data[['artists','popularity']].groupby(['artists']).mean().sort_values(['popularity'])
data[data['artists'] =="['Khalid']" ]
ar = data[['artists','popularity']].groupby(['artists']).mean().reset_index()
artist_p = {}
for p in ar.iterrows():
    #print(p[1][1])
    artist_p[p[1][0]] =  p[1][1]
    #break
    
data['artists_popularity'] = data['artists'].apply(lambda x: artist_p[x])
data.head()
X, y = data[['acousticness', 'danceability', 'duration_ms',
       'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 'loudness',
       'mode', 'speechiness', 'tempo', 'valence', 'artists_popularity',
          'year'
            ]], data[['popularity']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
reg1 = LinearRegression().fit(X_train, y_train)
print("Score on train:")
print("R^2", reg1.score(X_train, y_train))
print("MAE:", mean_absolute_error(y_train, reg1.predict(X_train)))

print()

print("Score on test:")
print("R^2", reg1.score(X_test,y_test))
print("MAE:", mean_absolute_error(y_test, reg1.predict(X_test)))
Y_predict = reg1.predict(X_test).astype('int32')
print("Accuracy score:", accuracy_score(y_test, Y_predict))
print("F1 score:", f1_score(y_test, Y_predict, average='micro'))
print("Recall score:", recall_score(y_test, Y_predict, average='micro'))
