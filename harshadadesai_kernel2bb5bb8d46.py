import itertools

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import pandas as pd

import numpy as np

import matplotlib.ticker as ticker

from sklearn import preprocessing

%matplotlib inline

!wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv

df = pd.read_csv('loan_train.csv')

df.head()
df.shape

df['due_date'] = pd.to_datetime(df['due_date'])

df['effective_date'] = pd.to_datetime(df['effective_date'])

df.head()

df['loan_status'].value_counts()

df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)

!conda install -c anaconda seaborn -y

import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)

g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)

g.map(plt.hist, 'Principal', bins=bins, ec="k")



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


df['Gender'] = df['Gender'].map({'male':0,'female':1})

df.head()
df['education'].unique()
df.groupby(['education'])['loan_status'].value_counts(normalize=True)
Feature = df[['Principal','terms','age','Gender','weekend']]

Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)

Feature.drop(['Master or Above'], axis = 1,inplace=True)

Feature.head()
X = Feature

X[0:5]
y = df['loan_status'].values

y[0:5]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)
from sklearn.neighbors import KNeighborsClassifier
k = 5

#Train Model and Predict  

KNN = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

KNN
y_pred =KNN.predict(X_test)

y_pred[0:5]
from sklearn.metrics import jaccard_score, log_loss, f1_score

print("Train set jaccard_similarity_score: ", jaccard_score(y_train, KNN.predict(X_train), pos_label='PAIDOFF'))

print("Test set jaccard_similarity_score: ", jaccard_score(y_test, y_pred, pos_label='PAIDOFF'))

print("Train set f1_score: ", f1_score(y_train, KNN.predict(X_train), average='weighted'))

print("Test set f1_score: ", f1_score(y_test, y_pred, average='weighted'))
from sklearn import tree
DT=tree.DecisionTreeClassifier()

DT.fit(X,y)
from sklearn.svm import SVC
svm=SVC(gamma='auto')

svm=svm.fit(X,y)
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression().fit(X, y)

from sklearn.metrics import jaccard_score

from sklearn.metrics import f1_score

from sklearn.metrics import log_loss

!wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv
test_df = pd.read_csv('loan_test.csv')

test_df.head()
df = test_df

df['due_date'] = pd.to_datetime(df['due_date'])

df['effective_date'] = pd.to_datetime(df['effective_date'])

df['dayofweek'] = df['effective_date'].dt.dayofweek

df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)

df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

df.groupby(['education'])['loan_status'].value_counts(normalize=True)

Feature = df[['Principal','terms','age','Gender','weekend']]

Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)

Feature.drop(['Master or Above'], axis = 1,inplace=True)

X_test = Feature

y_test = df['loan_status'].values

X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)
KNN_pred = KNN.predict(X_test)

DT_pred = DT.predict(X_test)

svm_pred = svm.predict(X_test)

LR_pred = LR.predict(X_test)

print("Train set jaccard_similarity_score: ", jaccard_score(y_test, KNN_pred,pos_label="PAIDOFF"))

print("Train set f1_score: ", f1_score(y_test, KNN_pred, average='weighted'))

jaccard_score = [jaccard_score(y_test, KNN_pred,pos_label="PAIDOFF"),

                 jaccard_score(y_test, DT_pred,pos_label="PAIDOFF"),

                 jaccard_score(y_test, svm_pred,pos_label="PAIDOFF"),

                 jaccard_score(y_test, LR_pred,pos_label="PAIDOFF")]
F1_score = [f1_score(y_test, KNN_pred, average='weighted'),

            f1_score(y_test, DT_pred, average='weighted'),

            f1_score(y_test, svm_pred, average='weighted'),

            f1_score(y_test, LR_pred, average='weighted')]

LR_pred
LR_pred_1 = (LR_pred == 'PAIDOFF')

y_test_1 = (y_test == 'PAIDOFF')

Logloss_score = ['NaN','NaN','NaN',log_loss(y_test_1, LR_pred_1)]
df = {'Algorithm': ['KNN', 'Decistion Tree', 'SVM', 'LogisticRegression'], \

     'Jaccard': jaccard_score, 'F1-score': F1_score, 'LogLoss': Logloss_score}

df_final = pd.DataFrame(data=df, columns=['Algorithm', 'Jaccard', 'F1-score', 'LogLoss'], index=None)

df_final
