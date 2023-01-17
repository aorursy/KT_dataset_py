import numpy as np 

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import seaborn as sns
#Machinelearning_imports

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/spotifyclassification/data.csv')
data.describe()
data.head()

train,test = train_test_split(data,test_size = 0.15)
train.shape
pos_tempo = data[data['target']==1]['tempo']

neg_tempo = data[data['target']==0]['tempo']



pos_acousticness = data[data['target']==1]['acousticness']

neg_acousticness = data[data['target']==0]['acousticness']



pos_danceability = data[data['target']==1]['danceability']

neg_danceability = data[data['target']==0]['danceability']



pos_duration_ms = data[data['target']==1]['duration_ms']

neg_duration_ms = data[data['target']==0]['duration_ms']



pos_energy = data[data['target']==1]['energy']

neg_energy = data[data['target']==0]['energy']





pos_instrumentalness = data[data['target']==1]['instrumentalness']

neg_instrumentalness = data[data['target']==0]['instrumentalness']



pos_key = data[data['target']==1]['key']

neg_key = data[data['target']==0]['key']



pos_mode = data[data['target']==1]['mode']

neg_mode = data[data['target']==0]['mode']





pos_liveness = data[data['target']==1]['liveness']

neg_liveness = data[data['target']==0]['liveness']



pos_loudness = data[data['target']==1]['loudness']

neg_loudness = data[data['target']==0]['loudness']



pos_speechiness = data[data['target']==1]['speechiness']

neg_speechiness = data[data['target']==0]['speechiness']



pos_time_signature = data[data['target']==1]['time_signature']

neg_time_signature = data[data['target']==0]['time_signature']



pos_valence = data[data['target']==1]['valence']

neg_valence = data[data['target']==0]['valence']

red_blue = ['#FF0000',

            '#00FFFF']

palette = sns.color_palette(red_blue)

sns.set_palette(palette)

sns.set_style('white')
fig = plt.figure(figsize=(12,12))

plt.title("Tempo")



pos_tempo.hist(alpha = 0.7, bins = 30, label = 'positive')

neg_tempo.hist(alpha = 0.7, bins = 30, label = 'negative')

plt.legend()

fig2 = plt.figure(figsize=(15,15))

ax1 = fig2.add_subplot(331)

ax1.set_title("Dance")

pos_danceability.hist(alpha = 0.7, bins = 30, label = 'positive')

neg_danceability.hist(alpha = 0.7, bins = 30, label = 'negative')

plt.legend()



ax1 = fig2.add_subplot(332)

ax1.set_title("Acousticness")

pos_acousticness.hist(alpha = 0.7, bins = 30, label = 'positive')

neg_acousticness.hist(alpha = 0.7, bins = 30, label = 'negative')

plt.legend()



ax1 = fig2.add_subplot(333)

ax1.set_title("Duration")

pos_duration_ms.hist(alpha = 0.7, bins = 30, label = 'positive')

neg_duration_ms.hist(alpha = 0.7, bins = 30, label = 'negative')

plt.legend()



ax1 = fig2.add_subplot(334)

ax1.set_title("Energy")

pos_energy.hist(alpha = 0.7, bins = 30, label = 'positive')

neg_energy.hist(alpha = 0.7, bins = 30, label = 'negative')

plt.legend()



ax1 = fig2.add_subplot(335)

ax1.set_title("instrumentalness")

pos_instrumentalness.hist(alpha = 0.7, bins = 30, label = 'positive')

neg_instrumentalness.hist(alpha = 0.7, bins = 30, label = 'negative')

plt.legend()



ax1 = fig2.add_subplot(336)

ax1.set_title("key")

pos_key.hist(alpha = 0.7, bins = 30, label = 'positive')

neg_key.hist(alpha = 0.7, bins = 30, label = 'negative')

plt.legend()



ax1 = fig2.add_subplot(337)

ax1.set_title("speechiness")

pos_speechiness.hist(alpha = 0.7, bins = 30, label = 'positive')

neg_speechiness.hist(alpha = 0.7, bins = 30, label = 'negative')

plt.legend()



ax1 = fig2.add_subplot(338)

ax1.set_title("Liveness")

pos_liveness.hist(alpha = 0.7, bins = 30, label = 'positive')

neg_liveness.hist(alpha = 0.7, bins = 30, label = 'negative')

plt.legend()



ax1 = fig2.add_subplot(339)

ax1.set_title("loudness")

pos_loudness.hist(alpha = 0.7, bins = 30, label = 'positive')

neg_loudness.hist(alpha = 0.7, bins = 30, label = 'negative')

plt.legend()
train = train.drop(['Unnamed: 0','song_title','artist'], axis = 1)

test = test.drop(['Unnamed: 0','song_title','artist'], axis = 1)
y_train = pd.DataFrame(train['target']) 

X_train = train.drop(['target'],axis = 1)



y_test = pd.DataFrame(test['target']) 

X_test = test.drop(['target'],axis = 1)

tree = DecisionTreeClassifier()

tree.fit(X_train,y_train)
ypred = tree.predict(X_test)

ypred
from sklearn.metrics import accuracy_score



accuracy_score(y_test,ypred)
from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(n_estimators=250)

forest.fit(X_train,y_train.values.ravel())

y_pred_forest = forest.predict(X_test)



accuracyforest = accuracy_score(y_test,y_pred_forest)

accuracyforest