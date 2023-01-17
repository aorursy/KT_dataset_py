import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter("ignore")
train = pd.read_csv('../input/train_data.csv')
train.head()
train.describe()
other_work = (train['workclass'] == '?') | (train['workclass'] == 'Without-pay') | (train['workclass'] == 'Never-worked')
train['workclass'][other_work] = 'other-workclass'
_ = train['workclass'].value_counts().plot(kind='bar')
_ = train['education'].value_counts().plot(kind='bar')
_ = train['marital.status'].value_counts().plot(kind='bar')
_ = train['occupation'].value_counts().plot(kind='bar')
_ = train['relationship'].value_counts().plot(kind='bar')
_ = train['race'].value_counts().plot(kind='bar')
_ = train['sex'].value_counts().plot(kind='bar')
train['native.country'][train['native.country'] != 'United-States'] = 'other_country'
_ = train['native.country'].value_counts().plot(kind='bar')
Y_train_1 = train['income']
#Y_train_1[Y_train_1 =='>50K'] = np.int16(1)
#Y_train_1[Y_train_1 =='<=50K'] = np.int16(0)

X_train_1 = pd.concat([train[['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']],
                  pd.get_dummies(train[['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']])], 
                  axis=1)
X_train_1.head()
X_train, X_test, Y_train, Y_test = train_test_split(X_train_1, Y_train_1, test_size=0.20, random_state=30)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
for i in range(3,12):
    clf = KNeighborsClassifier(n_neighbors = i, n_jobs=-1)
    print('K=' + str(i) + ': ' + str(np.mean(cross_val_score(clf, X_train_scaled, Y_train, cv=3, n_jobs=-1)))) 
clf = LogisticRegression()
cross_val_score(clf, X_train_scaled, Y_train, cv=3, n_jobs=-1)
clf = svm.SVC(gamma='auto')
cross_val_score(clf, X_train_scaled, Y_train, cv=3, n_jobs=-1)
for i in range(4, 12):
    for n in [50, 100, 250]:
        clf = RandomForestClassifier(n_estimators=n, max_features=i, n_jobs=-1)
        print('Features=' + str(i) + ", n=" + str(n) + ': ' + str(np.mean(cross_val_score(clf, X_train_scaled, Y_train, cv=5, n_jobs=-1)))) 
X_train_1_scaled = preprocessing.StandardScaler().fit_transform(X_train_1)

clf1 = KNeighborsClassifier(n_neighbors = 10, n_jobs=-1)
clf2 = LogisticRegression()
clf3 = svm.SVC(gamma='auto')
clf4 = RandomForestClassifier(n_estimators=250, max_features=11, n_jobs=-1)

clf = VotingClassifier(estimators=[('knn', clf1), ('logistic', clf2), ('svm', clf3), ('forest', clf4)], voting='hard')
clf.fit(X_train_1_scaled, Y_train_1)
#Y_test_pred = clf.predict(X_test)
#confusion_matrix(Y_test_pred, Y_test)
test = pd.read_csv('../input/test_data.csv')

other_work = (test['workclass'] == '?') | (test['workclass'] == 'Without-pay') | (test['workclass'] == 'Never-worked')
test['workclass'][other_work] = 'other-workclass'

test['native.country'][test['native.country'] != 'United-States'] = 'other_country'

X_test = pd.concat([test[['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']],
                  pd.get_dummies(test[['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']])], 
                  axis=1)
X_test.head()
X_test_scaled = preprocessing.StandardScaler().fit_transform(X_test)
Y_pred = clf.predict(X_test_scaled)
Y_pred