# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_master = pd.read_csv("/kaggle/input/parkinsons-data-set/parkinsons.data")

data_master.head(5)
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data_master.info()
fig,axes=plt.subplots(5,5,figsize=(15,15))

axes=axes.flatten()



for i in range(1,len(data_master.columns)-1):

    sns.boxplot(x='status',y=data_master.iloc[:,i],data=data_master,orient='v',ax=axes[i])

plt.tight_layout()

plt.show()
data_master.status.value_counts()
'''for dataset in data_master: 

    dataset['status'] = dataset['status'].astype(float) 



data_master['status'].value_counts()'''
X = data_master.drop(['status', 'name'], axis = 1)

y = data_master.status
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix



def clf_scores(clf, y_predicted):

    # Accuracy

    acc_train = clf.score(X_train, y_train)*100

    acc_test = clf.score(X_test, y_test)*100

    

    roc = roc_auc_score(y_test, y_predicted)*100 

    tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()

    cm = confusion_matrix(y_test, y_predicted)

    correct = tp + tn

    incorrect = fp + fn

    

    return acc_train, acc_test, roc, correct, incorrect, cm
#1. Logistic regression



from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression()

clf_lr.fit(X_train, y_train)



Y_pred_lr = clf_lr.predict(X_test)

print(clf_scores(clf_lr, Y_pred_lr))
#2. KNN



from sklearn.neighbors import KNeighborsClassifier

clf_knn = KNeighborsClassifier(n_neighbors=3)

clf_knn.fit(X_train, y_train)



Y_pred_knn = clf_knn.predict(X_test)

print(clf_scores(clf_knn, Y_pred_knn))
#3. Naive Bayes



from sklearn.naive_bayes import GaussianNB

clf_gnb = GaussianNB()

clf_gnb.fit(X_train, y_train)



Y_pred_gnb = clf_gnb.predict(X_test)

print(clf_scores(clf_gnb, Y_pred_gnb))
data_master.columns
#Scaling

from sklearn.preprocessing import StandardScaler



# copy of datasets

X_train_scaled = X_train.copy()

X_test_scaled = X_test.copy()



# numerical features

num_cols = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',

       'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',

       'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',

       'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',

       'spread1', 'spread2', 'D2', 'PPE']



# apply standardization on numerical features

for i in num_cols:

    

    # fit on training data column

    scale = StandardScaler().fit(X_train_scaled[[i]])

    

    # transform the training data column

    X_train_scaled[i] = scale.transform(X_train_scaled[[i]])

    

    # transform the testing data column

    X_test_scaled[i] = scale.transform(X_test_scaled[[i]])

X_train.describe()
X_train_scaled.describe()
#1. Logistic regression



from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression()

clf_lr.fit(X_train_scaled, y_train)



Y_pred_lr = clf_lr.predict(X_test_scaled)

print(clf_scores(clf_lr, Y_pred_lr))
#2. KNN



from sklearn.neighbors import KNeighborsClassifier

clf_knn = KNeighborsClassifier(n_neighbors=3)

clf_knn.fit(X_train_scaled, y_train)



Y_pred_knn = clf_knn.predict(X_test_scaled)

print(clf_scores(clf_knn, Y_pred_knn))
#3. Naive Bayes



from sklearn.naive_bayes import GaussianNB

clf_gnb = GaussianNB()

clf_gnb.fit(X_train_scaled, y_train)



Y_pred_gnb = clf_gnb.predict(X_test_scaled)

print(clf_scores(clf_gnb, Y_pred_gnb))
#4. SVM



from sklearn.svm import SVC



clf_svm = SVC()

clf_svm.fit(X_train_scaled, y_train)



Y_pred_svm = clf_svm.predict(X_test_scaled)

print(clf_scores(clf_svm, Y_pred_svm))
#Meta-classifier

from mlxtend.classifier import StackingClassifier

from sklearn import model_selection



lr = LogisticRegression()

sclf = StackingClassifier(classifiers=[clf_knn, clf_svm, clf_gnb], 

                          meta_classifier=lr)

sclf.fit(X_train_scaled, y_train)

for clf, label in zip([clf_knn, clf_svm, clf_gnb, sclf], 

                      ['KNN', 

                       'SVM', 

                       'Naive Bayes',

                       'StackingClassifier']):



    Y_pred = clf.predict(X_test_scaled)

    scores = clf_scores(clf, Y_pred)

    #scores = model_selection.cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='accuracy')

    

    print(scores, label)