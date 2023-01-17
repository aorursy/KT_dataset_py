#Imports
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
orig = pd.read_csv('../input/mushrooms.csv')
#Shuffles the orig DataFrame
#orig = orig.sample(frac=1)
orig.head()
X = orig.drop(['class'], axis=1)
y = orig['class']
for attr in X.columns:
    print('\n*', attr, '*')
    print(X[attr].value_counts())
X.drop(['veil-type'], axis=1, inplace=True)
for attr in X.columns:
    fig, ax =plt.subplots(1,2)
    sns.countplot(X[X['stalk-root']=='?'][attr], ax=ax[0]).set_title('stalk-root = ?')
    sns.countplot(X[X['stalk-root']!='?'][attr], ax=ax[1]).set_title('stalk-root != ?')
    fig.show()
#For columns with only two values
for col in X.columns:
    if len(X[col].value_counts()) == 2:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
X.head()
X = pd.get_dummies(X)
X.head()
#New
#train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.0)
#New (used to show train_X is indeed the same as X, albeit, shuffled)
#print(len(X))
#print(len(train_X), len(train_y))
#print(len(val_X), len(val_y))
kmeans = KMeans(n_clusters=2, random_state=None)

#Old
kmeans.fit(X)

#New
#kmeans.fit(train_X)
#Old
clusters = kmeans.predict(X)

#New
#clusters = kmeans.predict(train_X)
clusters
cluster_df = pd.DataFrame()
cluster_df['cluster'] = clusters

#Old
cluster_df['class'] = y

#New
#cluster_df['class'] = train_y
sns.factorplot(col='cluster', y=None, x='class', data=cluster_df, kind='count')