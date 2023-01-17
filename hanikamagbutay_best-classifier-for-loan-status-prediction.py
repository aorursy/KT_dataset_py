import itertools

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

from matplotlib.ticker import NullFormatter

from sklearn import preprocessing

%matplotlib inline
data = pd.read_csv('/kaggle/input/loandata/Loan payments data.csv')
data.head()
data.dtypes
data['effective_date'] = pd.to_datetime(data['effective_date'])

data['due_date'] = pd.to_datetime(data['due_date'])

data = data.drop('paid_off_time', 1).drop('past_due_days', 1).drop('Loan_ID', 1)

data.head()
data.isnull().sum()
data['loan_status'].value_counts()
bins = np.linspace(data.Principal.min(), data.Principal.max(), 10)

g = sns.FacetGrid(data, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)

g.map(plt.hist, 'Principal', bins=bins, ec="k")



g.axes[-1].legend()

plt.show()
bins = np.linspace(data.age.min(), data.age.max(), 10)

g = sns.FacetGrid(data, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)

g.map(plt.hist, 'age', bins=bins, ec="k")



g.axes[-1].legend()

plt.show()
data['day'] = data['effective_date'].dt.dayofweek

bins = np.linspace(data.day.min(), data.day.max(), 10)

g = sns.FacetGrid(data, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)

g.map(plt.hist, 'day', bins=bins, ec="k")

g.axes[-1].legend()

plt.show()
data['weekend'] = data['day'].apply(lambda x: 1 if (x>3)  else 0)

data.head()
data['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
data.groupby(['education'])['loan_status'].value_counts()
features = data[['Principal','terms','age','Gender','weekend','education']]

features.head()
features = pd.concat([features,pd.get_dummies(data['education'])], axis=1)

features = features.drop(['education'], axis=1).drop(['Master or Above'], axis = 1)

features.head()
X = features

X.head()
y = data['loan_status'].values

y[0:5]
X = preprocessing.StandardScaler().fit(X).transform(X)

X[0:5]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics



#Get best K

Ks = 11

mean_acc = np.zeros((Ks-1))

std_acc = np.zeros((Ks-1))

ConfusionMx = [];

for n in range(1,Ks):

    

    #Train Model and Predict  

    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)

    yhat=neigh.predict(X_test)

    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)



    

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])



mean_acc
plt.plot(range(1,Ks),mean_acc,'g')

plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)

plt.ylabel('Accuracy ')

plt.xlabel('Number of Neighbors (K)')

plt.tight_layout()

plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k =", mean_acc.argmax()+1)
k = 8



knn = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)

yhat = knn.predict(X_test)
print("Train Set Accuracy: ", metrics.accuracy_score(y_train, knn.predict(X_train)))

print("Test Set Accuracy: ", metrics.accuracy_score(y_test, yhat))
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

tree.fit(X_train,y_train)

tree
yhat = tree.predict(X_test)

yhat
print("Train Set Accuracy: ", metrics.accuracy_score(y_train, tree.predict(X_train)))

print("Test Set Accuracy: ", metrics.accuracy_score(y_test, yhat))
from sklearn import svm

clf = svm.SVC(kernel='rbf')

clf.fit(X_train, y_train)
yhat = clf.predict(X_test)

yhat
print("Train Set Accuracy: ", metrics.accuracy_score(y_train, clf.predict(X_train)))

print("Test Set Accuracy: ", metrics.accuracy_score(y_test, yhat))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

LR
yhat = LR.predict(X_test)

yhat
print("Train Set Accuracy: ", metrics.accuracy_score(y_train, LR.predict(X_train)))

print("Test Set Accuracy: ", metrics.accuracy_score(y_test, yhat))
from sklearn.metrics import jaccard_score

from sklearn.metrics import f1_score

from sklearn.metrics import log_loss
test_df = pd.read_csv('/kaggle/input/loandata/Loan payments data.csv')

test_df.head()
# Preprocessing



test_df['due_date'] = pd.to_datetime(test_df['due_date'])

test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])

test_df['day'] = test_df['effective_date'].dt.dayofweek

test_df['weekend'] = test_df['day'].apply(lambda x: 1 if (x>3)  else 0)

test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

test_feature = test_df[['Principal','terms','age','Gender','weekend']]

test_feature = pd.concat([test_feature,pd.get_dummies(test_df['education'])], axis=1)

test_feature.drop(['Master or Above'], axis = 1,inplace=True)

test_X = preprocessing.StandardScaler().fit(test_feature).transform(test_feature)

test_y = test_df['loan_status'].values
# KNN

knn_yhat = knn.predict(test_X)

ji1 = round(jaccard_score(test_y, knn_yhat, average='weighted'),2)

# Decision Tree

dt_yhat = tree.predict(test_X)

ji2 = round(jaccard_score(test_y, dt_yhat, average='weighted'),2)

# SVM

svm_yhat = clf.predict(test_X)

ji3 = round(jaccard_score(test_y, svm_yhat, average='weighted'),2)

# Logistic Regression

lr_yhat = LR.predict(test_X)

ji4 = round(jaccard_score(test_y, lr_yhat, average='weighted'),2)



list_ji = [ji1, ji2, ji3, ji4]

list_ji
# KNN

fs1 = round(f1_score(test_y, knn_yhat, average='weighted'),2)

# Decision Tree

fs2 = round(f1_score(test_y, dt_yhat, average='weighted'),2)

# SVM

fs3 = round(f1_score(test_y, svm_yhat, average='weighted'),2)

# Logistic Regression

fs4 = round(f1_score(test_y, lr_yhat, average='weighted'),2)



list_fs = [fs1, fs2, fs3, fs4]

list_fs
accuracy = pd.DataFrame(list_ji, index=['KNN','Decision Tree','SVM','Logistic Regression'])

accuracy.columns = ['Jaccard']

accuracy.insert(loc=1, column='F1-score', value=list_fs)

accuracy.columns.name = 'Algorithm'

accuracy