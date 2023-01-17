import itertools

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import pandas as pd

import numpy as np

import matplotlib.ticker as ticker

from sklearn import preprocessing

import sklearn

%matplotlib inline
#!wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv
df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')

df.head()
df.shape
df['due_date'] = pd.to_datetime(df['due_date'])

df['effective_date'] = pd.to_datetime(df['effective_date'])

df.head()
df['loan_status'].value_counts()
# notice: installing seaborn might takes a few minutes

#!conda install -c anaconda seaborn -y
import seaborn as sns



bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)

g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)

g.map(plt.hist, 'Principal', bins=bins, ec="k")



g.axes[-1].legend()

plt.show()
bins = np.linspace(df.age.min(), df.age.max(), 10)

g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)

g.map(plt.hist, 'age', bins=bins, ec="k")



g.axes[-1].legend()

plt.show()
df['dayofweek'] = df['effective_date'].dt.dayofweek

bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)

g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)

g.map(plt.hist, 'dayofweek', bins=bins, ec="k")

g.axes[-1].legend()

plt.show()

df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

df.head()
df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

df.head()

df.groupby(['education'])['loan_status'].value_counts(normalize=True)
df[['Principal','terms','age','Gender','education']].head()
Feature = df[['Principal','terms','age','Gender','weekend']]

Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)

Feature.drop(['Master or Above'], axis = 1,inplace=True)

Feature.head()

X = Feature

X[0:5]
y = df['loan_status'].values

y[0:5]
X= preprocessing.StandardScaler().fit(X).transform(X)

X[0:5]
from sklearn.neighbors import KNeighborsClassifier

# Create KNN classifier

knn = KNeighborsClassifier(n_neighbors = 5)

# Fit the classifier to the data

knn.fit(X,y)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = "entropy", splitter = "random", max_depth = 2,  min_samples_split = 5,

                              min_samples_leaf = 2, max_features = 2)
tree.fit(X,y)
from sklearn.svm import SVC

sv = SVC(gamma='auto',probability=True)
sv.fit(X, y)
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(max_iter=1000)
logisticRegr.fit(X,y)
from sklearn.metrics import jaccard_similarity_score

from sklearn.metrics import f1_score

from sklearn.metrics import log_loss
#!wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv


def testFunc ( preds,predsprob,y_test ):

    f1 = f1_score(y_test,preds,pos_label='PAIDOFF')

    ll = log_loss(y_test,predsprob)

    #j_index = jaccard_similarity_score(y_true=y_test,y_pred=preds)

    j_index =  sklearn.metrics.jaccard_score(y_test, preds, labels=None, pos_label="PAIDOFF", average='binary', sample_weight=None)



    j_index = round(j_index,2)

    ll = round(ll,2)

    f1 = round(f1,2)

    

    print('F1_Score = '+str(f1)+',LogLoss='+str(ll)+',jaccard_score='+str(j_index))

   



test_df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')

test_df.head()
test_df['due_date'] = pd.to_datetime(test_df['due_date'])

test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])

test_df.head()



test_df['loan_status'].value_counts()





test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek

bins = np.linspace(test_df.dayofweek.min(), test_df.dayofweek.max(), 10)

g = sns.FacetGrid(test_df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)

g.map(plt.hist, 'dayofweek', bins=bins, ec="k")

g.axes[-1].legend()

plt.show()





test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

test_df.head()



test_df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)





test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

test_df.head()





test_df.groupby(['education'])['loan_status'].value_counts(normalize=True)



test_df[['Principal','terms','age','Gender','education']].head()







test_feature = test_df[['Principal','terms','age','Gender','weekend']]

test_feature = pd.concat([test_feature,pd.get_dummies(test_df['education'])], axis=1)

test_feature.drop(['Master or Above'], axis = 1,inplace=True)

test_feature.head()



X_test = test_feature

X_test[0:5]





y_test = test_df['loan_status'].values

y_test[0:5]



X_test= preprocessing.StandardScaler().fit(X_test).transform(X_test)





X_test[0:5]
knn_preds = knn.predict(X_test)

knn_predsprob = knn.predict_proba(X_test)

print("KNN")

testFunc(knn_preds,knn_predsprob,y_test)
import warnings

warnings.filterwarnings('always')  



tree_preds = tree.predict(X_test)

tree_predsprob = tree.predict_proba(X_test)



f1 = f1_score(y_test,tree_preds,average='weighted', labels=np.unique(tree_preds))

ll = log_loss(y_test,tree_predsprob)

j_index =  sklearn.metrics.jaccard_score(y_test, tree_preds, labels=None, pos_label="PAIDOFF", average='binary', sample_weight=None)





j_index = round(j_index,2)

ll = round(ll,2)

f1 = round(f1,2)



print("Decesion Tree")

print('F1_Score = '+str(f1)+',LogLoss='+str(ll)+',jaccard_score='+str(j_index))


sv_preds = sv.predict(X_test)

sv_predsprob = sv.predict_proba(X_test)





f1 = f1_score(y_test,sv_preds,average='weighted', labels=np.unique(sv_preds))

ll = log_loss(y_test,sv_predsprob)

j_index = sklearn.metrics.jaccard_score(y_test, sv_preds, labels=None, pos_label="PAIDOFF", average='binary', sample_weight=None)





j_index = round(j_index,2)

ll = round(ll,2)

f1 = round(f1,2)





print("Service Vector machine ")

print('F1_Score = '+str(f1)+',LogLoss='+str(ll)+',jaccard_score='+str(j_index))







logisticRegr_preds = logisticRegr.predict(X_test)

logisticRegr_predsprob = logisticRegr.predict_proba(X_test)



print("Logistic Regression")

testFunc(logisticRegr_preds,logisticRegr_predsprob,y_test)