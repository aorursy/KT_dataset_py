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
df_train = pd.read_csv('/kaggle/input/novartis-data/Train.csv')

df_eval = pd.read_csv('/kaggle/input/novartis-data/Test.csv')

df_train.head()
X = df_train.drop(['MULTIPLE_OFFENSE', 'DATE', 'INCIDENT_ID'], axis=1)

eval_X = df_eval.drop(['DATE', 'INCIDENT_ID'], axis=1)

Y = df_train['MULTIPLE_OFFENSE']



incident_ids_train = df_train['INCIDENT_ID']

incident_ids_eval = df_eval['INCIDENT_ID']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.33)



print(X_train.shape, y_train.shape)

print(X_cv.shape, y_cv.shape)

print(X_test.shape, y_test.shape)
from sklearn.preprocessing import StandardScaler



## Replacing NA with zeros

X_train.fillna(0, inplace=True)

X_cv.fillna(0, inplace=True)

X_test.fillna(0, inplace=True)

eval_X.fillna(0, inplace=True)



## Standarization

scaler = StandardScaler()

scaler.fit(X_train)



X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)

X_cv = pd.DataFrame(scaler.transform(X_cv), columns=X_cv.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

eval_X = pd.DataFrame(scaler.transform(eval_X), columns=eval_X.columns)
Y.value_counts()
## Increasing the number of negative cases 

## OR use class labels or class weights while fitting models



from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt



train_auc = []

cv_auc = []

k = [1, 3, 5, 7, 9, 11, 13]



for i in k:

    neigh = KNeighborsClassifier(n_neighbors=i)

    neigh.fit(X_train, y_train)

    

    y_train_pred = neigh.predict_proba(X_train)[:,1]

    y_cv_pred = neigh.predict_proba(X_cv)[:,1]

    

    train_auc.append(roc_auc_score(y_train, y_train_pred))

    cv_auc.append(roc_auc_score(y_cv, y_cv_pred))

    

plt.plot(k, train_auc, label='Train AUC')

plt.plot(k, cv_auc, label='CV AUC')

plt.legend()

plt.xlabel('k: Hyperparameter')

plt.ylabel('AUC')

plt.title('ERROR Plots')

plt.show()
from sklearn.metrics import roc_curve, auc

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import svm



best_k = 13 # from above curve



## k-NN

# neigh = KNeighborsClassifier(n_neighbors=best_k)



## Gaussian Naive Baye's (score = 75)

# neigh = GaussianNB()



## SVM (score = 90)

## Assigning a class weight to 0 class as the dataset is unbiased

# neigh = svm.SVC(probability=True, class_weight={0: 10})



## RandomForest (score = 93.5)

# neigh = RandomForestClassifier(n_estimators=100, criterion='entropy')



## XG Boost 

neigh = GradientBoostingClassifier()



neigh.fit(X_train, y_train)



train_fpr, train_tpr, thresholds = roc_curve(y_train, neigh.predict_proba(X_train)[:,1])

test_fpr, test_tpr, thresholds = roc_curve(y_test, neigh.predict_proba(X_test)[:,1])



plt.plot(train_fpr, train_tpr)

plt.plot(test_fpr, test_tpr)

plt.legend()

plt.xlabel('K: Hyperparameter')

plt.ylabel('AUC')

plt.title('Error Plots')

plt.show()
from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score



y_train_predict = neigh.predict(X_train)

y_test_predict = neigh.predict(X_test)



train_confusion_matrix = confusion_matrix(y_train, y_train_predict)

test_confusion_matrix = confusion_matrix(y_test, y_test_predict)

print("Train Confusion Matrix")

print(train_confusion_matrix)

print("Test Confusion Matrix")

print(test_confusion_matrix)



print("Training F1 Score")

print(f1_score(y_train, y_train_predict))

print("Test F1 Score")

print(f1_score(y_test, y_test_predict))

print(eval_X.shape)

eval_X.head()
results = neigh.predict(eval_X)

results_df = pd.DataFrame({ 'MULTIPLE_OFFENSE': results, 'INCIDENT_ID': incident_ids_eval })

results_df = results_df[['INCIDENT_ID', 'MULTIPLE_OFFENSE']]

print(results_df.shape)

results_df.head()
results_df.to_csv('submission.csv', index=False)