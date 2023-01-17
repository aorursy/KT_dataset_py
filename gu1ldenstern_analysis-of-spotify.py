# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
spotify = pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data.csv')
spotify.head()
print(len(spotify))
import seaborn as sns
ax = sns.barplot(x="mode", y="valence", data=spotify)
ax.set()
ax = sns.barplot(x="mode", y="explicit", data=spotify)
ax.set()
beatles_n_stones = spotify.loc[lambda spotify: (spotify['artists'] == "['The Beatles']")|(spotify['artists'] == "['The Rolling Stones']")]
beatles_n_stones = beatles_n_stones[beatles_n_stones.year<1970]
import copy
checkpoint_bns = copy.deepcopy(beatles_n_stones)
print(len(beatles_n_stones), 275+263, sep = ' ')
beatles_n_stones.tail()
del beatles_n_stones['id']
del beatles_n_stones ['release_date']
del beatles_n_stones ['year']
del beatles_n_stones ['name']
beatles_n_stones = beatles_n_stones.iloc[:, 1:] #дада мне лень вспоминать как называется тот столбец

beatles_n_stones = beatles_n_stones.replace("['The Beatles']", 1)
beatles_n_stones = beatles_n_stones.replace("['The Rolling Stones']", 0)
B_n_S = ['Rolling Stones', 'Beatles']
Y = beatles_n_stones.iloc[:, 1].values
del beatles_n_stones ['artists']
X = beatles_n_stones.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
#print(X_test[1])
finding_the_name = {i:X_test[i, 2] for i in range(len(X_test))}
finding_the_name
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from matplotlib import pyplot
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=200)#, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

my_features = list(beatles_n_stones)
importance = classifier.feature_importances_
for i,v in enumerate(importance):
    print('Feature: %0s, Score: %.5f' % (my_features[i],v))

pyplot.bar([my_features[x][:3] for x in range(len(importance))], importance)
pyplot.show()

beatles = spotify.loc[lambda spotify: (spotify['artists'] == "['The Beatles']")]
ax = sns.countplot(x="key", hue="mode", data=beatles)
ax = sns.barplot(x="year", y="valence",palette="rocket",  data=beatles.loc[beatles['year']<1970])

sns.set(rc={'figure.figsize':(8,3)})
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
from sklearn.metrics import plot_precision_recall_curve
disp = plot_precision_recall_curve(classifier, X_test, y_test)
from sklearn.metrics import plot_roc_curve
disp = plot_roc_curve(classifier, X_test, y_test)
for row_index, (input, prediction, label) in enumerate(zip (X_test, y_pred, y_test)):
  if prediction != label:
    length = finding_the_name[row_index]
    name = checkpoint_bns.loc[checkpoint_bns['duration_ms'] == length].iloc[0, 13]
    print(name, 'был классифицирован как', B_n_S[prediction], 'а должен был быть классифицирован как', B_n_S[label])