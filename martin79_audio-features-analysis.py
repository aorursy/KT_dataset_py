import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import mutual_info_classif

from sklearn.model_selection import train_test_split  

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report,confusion_matrix
class_names=['dinner','party','sleep','workout']

dfs = []

for i in range(len(class_names)):

    dfs.append(pd.read_csv('../input/'+class_names[i]+'_track.csv'))

    dfs[i]['class'] = i + 1

    dfs[i]['class_names'] = class_names[i]
df = pd.concat(dfs)
df.describe()
df.dropna().describe()
df = df.dropna()

df.head()
df.columns
numerical_features = ['acousticness', 'danceability',

       'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness',

       'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']
df.shape
X = df[numerical_features]

y = df['class']

X.shape
mi = mutual_info_classif(X, y)

print(mi)
sns.pairplot(df,vars= ['loudness','acousticness','energy','instrumentalness'],hue='class_names')
sc = StandardScaler()

Xsc = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(Xsc, y, test_size=0.2, random_state=42)
rfc = RandomForestClassifier(1000)
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

print("accuracy: %.1f%%" % (np.mean(y_test==y_pred)*100))
class_names=['dinner','party','sleep','workout']

dfs = []

for i in range(len(class_names)):

    dfs.append(pd.read_csv('../input/'+class_names[i]+'_audio.csv'))

    dfs[i]['class'] = i + 1

    dfs[i]['class_names'] = class_names[i]
df = pd.concat(dfs)
df.describe()
df.head()
X = df[['mfcc','scem','scom','srom','sbwm','tempo','rmse']]

y = df['class']
mi = mutual_info_classif(X, y)

print(mi)
sns.pairplot(df,vars= ['mfcc','scem','scom','srom','sbwm','rmse'],hue='class_names')
print(X.corr())
Xsc = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(Xsc, y, test_size=0.3, random_state=42)
X_train.shape
X_test.shape
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

print("accuracy: %.1f %%"%(np.mean(y_test==y_pred)*100))