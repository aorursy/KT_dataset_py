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

df.head().style.background_gradient(cmap='RdGy')
df.shape
df['due_date'] = pd.to_datetime(df['due_date'])

df['effective_date'] = pd.to_datetime(df['effective_date'])

df.head().style.background_gradient(cmap='RdGy')
df['loan_status'].value_counts()
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

df.head().style.background_gradient(cmap='RdGy')
df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

df.head().style.background_gradient(cmap='RdGy')
df.groupby(['education'])['loan_status'].value_counts(normalize=True)
df[['Principal','terms','age','Gender','education']].head().style.background_gradient(cmap='RdGy')
Feature = df[['Principal','terms','age','Gender','weekend']]

Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)

Feature.drop(['Master or Above'], axis = 1,inplace=True)

Feature.head().style.background_gradient(cmap='RdGy')

X = Feature

X1 = X

X[0:5]
print(X.columns)
y = df['loan_status'].values

y[0:5]
X= preprocessing.StandardScaler().fit(X).transform(X)

X[0:5]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

Ks = 10

mean_acc = np.zeros((Ks-1))

std_acc = np.zeros((Ks-1))

ConfustionMx = [];

for n in range(1,Ks):

    #Train Model and Predict  

    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)

    yhat=neigh.predict(X_test)

    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

#Plot model accuracy for Different number of Neighbors

plt.plot(range(1,Ks),mean_acc,'g')

plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)

plt.legend(('Accuracy ', '+/- 3xstd'))

plt.ylabel('Accuracy ')

plt.xlabel('Number of Neighbors (K)')

plt.tight_layout()

plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

k = 7 # For best K

#Train Model and Predict  

neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

yhat = neigh.predict(X_test)

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))

print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

drugTree # it shows the default parameters

drugTree.fit(X_train,y_train)

predTree = drugTree.predict(X_test)



from sklearn import metrics

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))
#INSTALLATIONS TO VIEW THE DECISION TREE

!conda install -c conda-forge pydotplus -y
from sklearn.externals.six import StringIO

import pydotplus

import matplotlib.image as mpimg

from sklearn import tree



%matplotlib inline 

dot_data = StringIO()

filename = "drugtree.png"

featureNames = df.columns[3: 11]

targetNames = df["loan_status"].unique().tolist()

out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)  

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png(filename)

img = mpimg.imread(filename)

plt.figure(figsize=(100, 200))

plt.imshow(img,interpolation='nearest')
from sklearn import svm

from sklearn import metrics

clf2 = svm.SVC(kernel='linear')

clf2.fit(X_train, y_train) 

yhat2 = clf2.predict(X_test)

print("SVM ", metrics.accuracy_score(y_test, yhat2))
clf2.support_vectors_
print(X1.columns)
# Determining the most contributing features for SVM classifier



pd.Series(abs(clf2.coef_[0]), index=X1.columns).nlargest(10).plot(kind='barh',figsize=(8, 6))
clf2 = svm.SVC(kernel='rbf')

clf2.fit(X_train, y_train) 

yhat2 = clf2.predict(X_test)

print("SVM ", metrics.accuracy_score(y_test, yhat2))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import log_loss

from sklearn.metrics import jaccard_similarity_score

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

yhat = LR.predict(X_test)

yhat_prob = LR.predict_proba(X_test)

print("Logistic Regression's Log Loss Accuracy: ", log_loss(y_test, yhat_prob))

print("Logistic Regression's Jaccard Similarity Accuracy: ", jaccard_similarity_score(y_test, yhat))

from sklearn.metrics import jaccard_similarity_score

from sklearn.metrics import f1_score

from sklearn.metrics import log_loss
!wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv
test_df = pd.read_csv('loan_test.csv')

test_df.head().style.background_gradient(cmap='RdGy')
# Preprocessing the test data set

test_df['due_date'] = pd.to_datetime(test_df['due_date'])

test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])

test_df['loan_status'].value_counts()

test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek

test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

test_df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)

test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

test_df.groupby(['education'])['loan_status'].value_counts(normalize=True)

test_df[['Principal','terms','age','Gender','education']].head()

test_df[['Principal','terms','age','Gender','education']].head()

Feature = test_df[['Principal','terms','age','Gender','weekend']]

Feature = pd.concat([Feature,pd.get_dummies(test_df['education'])], axis=1)

Feature.drop(['Master or Above'], axis = 1,inplace=True)

X = Feature

y_test = test_df['loan_status'].values

X_test= preprocessing.StandardScaler().fit(X).transform(X)
# K Nearest Neighbor(KNN) Prediction

yhat = neigh.predict(X_test)

print("KNN's Jaccard Similarity Accuracy: %.2f" % jaccard_similarity_score(y_test, yhat))

print("KNN Avg F1-score: %.2f" % f1_score(y_test, yhat, average='weighted'))
# Support Vector Machine Prediction

yhat2 = clf2.predict(X_test)

print("SVM's Jaccard Similarity Accuracy: %.2f" % jaccard_similarity_score(y_test, yhat2))

print("SVM Avg F1-score: %.2f" % f1_score(y_test, yhat2, average='weighted'))

# Logistic Regression Prediction

yhat3 = LR.predict(X_test)

yhat_prob3 = LR.predict_proba(X_test)

print("Logistic Regression's Jaccard Similarity Accuracy:  %.2f" % jaccard_similarity_score(y_test, yhat3))

print("LRP Avg F1-score: %.2f" % f1_score(y_test, yhat3, average='weighted'))

print("Logistic Regression's Log Loss Accuracy: %.2f" % log_loss(y_test, yhat_prob3))
# Decision Tree Prediction

predTree = drugTree.predict(X_test)

print("DecisionTrees's Jaccard Similarity Accuracy: %.2f" % jaccard_similarity_score(y_test, predTree))

print("DecisionTrees Avg F1-score: %.2f" % f1_score(y_test, predTree, average='weighted'))