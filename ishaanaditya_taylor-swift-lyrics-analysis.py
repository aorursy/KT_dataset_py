import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

sns.set(context='notebook', style='darkgrid') 
df=pd.read_csv("../input/taylorswiftlyricsfeatures/TaylorSwiftLyricsFeatureSet.csv")
df=df.rename(columns={"Album":"track_album","Artist":"track_artist","Track":"track_title","TrackURI":"track_uri" , "TrackID":"track_id" ,"Lyrics":"track_lyric"})
data = df.drop(columns=["track_lyric"],axis=0)
data.head()
XtoPredict = data.loc[data['genres'].isnull(),:].drop(columns=['track_uri','track_id']).reset_index(drop=True)
FeatureSet = data.loc[data['genres'].notnull(),:].drop(columns=['track_uri','track_id']).reset_index(drop=True)
FeatureSet.head()
print(len(FeatureSet.genres.unique()))

country = ['Country Pop','Country','Folk Pop','Blue grass','Contemporary Country']
pop = ['Pop','Electropop','Synth Pop','Dance Pop','Dream Pop']
rock = ['Pop Rock','Pop Punk','Alternative Rock','Soft Rock','R&B','Country Rock',]

print(len(rock)+len(country)+len(pop))
genre_broad = []
for index,i in enumerate(FeatureSet.genres):
    if i in country:
        genre_broad.append('country')
    
    if i in rock:
        genre_broad.append('rock')
    
    if i in pop:
        genre_broad.append('pop')
        
FeatureSet['genre_broad']=genre_broad
FeatureSet.groupby('genre_broad').count()['track_title'].plot.bar()
fig = plt.figure(figsize=(7,5))
sns.boxplot(x='duration_ms',data=FeatureSet)
plt.title("Duration (to check outliers)")
fig = plt.figure(figsize=(15,15))

corr =df.loc[:,'danceability':'time_signature'].corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool)) #For Lower Triangle, removes TriU

sns.heatmap(corr,annot=True,mask=mask,cmap='RdBu')
genre_rel= FeatureSet.groupby('genres').median().loc[:,'danceability':'time_signature']

corr=genre_rel.transpose().corr('kendall')
mask = np.triu(np.ones_like(corr, dtype=np.bool)) #For Lower Triangle, removes TriU

fig = plt.figure(figsize=(15,15))
sns.heatmap(corr,annot=True,mask=mask,cmap='RdBu')
genre_rel= FeatureSet.groupby('genre_broad').median().loc[:,'danceability':'time_signature']

corr=genre_rel.transpose().corr('kendall')
#mask = np.triu(np.ones_like(corr, dtype=np.bool)) #For Lower Triangle, removes TriU

fig = plt.figure(figsize=(5,5))
sns.heatmap(corr,annot=True,cmap='RdBu')
fig = plt.figure(figsize=(15,7))

fig.add_subplot(2,4,1)
sns.distplot(data.danceability)

fig.add_subplot(2,4,2)
sns.distplot(data.energy)

fig.add_subplot(2,4,3)
sns.distplot(data.key)

fig.add_subplot(2,4,4)
sns.distplot(data.loudness)

fig.add_subplot(2,4,5)
sns.distplot(data.speechiness)

fig.add_subplot(2,4,6)
sns.distplot(data.acousticness)

fig.add_subplot(2,4,7)
sns.distplot(data.valence)

fig.add_subplot(2,4,8)
sns.distplot(data.tempo)
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
X = FeatureSet.loc[:,'danceability':'duration_ms'].drop(columns=['loudness','mode'])
y = FeatureSet.loc[:,'genre_broad']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.25)
dt = DecisionTreeClassifier(min_samples_leaf=1)
dt.fit(X_train, y_train)

fig = plt.figure(figsize=(25,10))
#tree.plot_tree(dt);

a = plot_tree(dt, 
              feature_names=X.columns, 
              class_names=y.unique(), 
              label={"root"},
              proportion=True,
              filled=True, 
              impurity=False,
              rounded=True, 
              fontsize=15)

yhat=dt.predict(X_test)
dt.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(criterion='gini',
                             n_estimators=100)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print(gnb.score(X_test, y_test))

mnb = MultinomialNB(alpha=1000)
print((mnb.fit(X_train, y_train)).score(X_test, y_test))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_test,y_test)