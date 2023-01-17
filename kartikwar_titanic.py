#Import libraries
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_curve, auc

from sklearn.decomposition import PCA



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Preprocess Data
train_path = os.path.abspath("../input/train.csv")

train = pd.read_csv(train_path)

test_path = os.path.abspath("../input/test.csv")

test = pd.read_csv(test_path)





X_train = train.drop('Survived', axis=1)

X_test = test

y_train = train['Survived']







X_train['Age'] = X_train['Age'].apply(str)

X_test['Age'] = X_test['Age'].apply(str)

X_test['Fare'] = X_test['Fare'].apply(str)





for col in X_train.columns:

    label_encoder = LabelEncoder()

    try:

        X_train[col] = X_train[col].fillna("")

        X_train[col] = label_encoder.fit_transform(X_train[col])

    except Exception as e :

        print(col)

        print(e)

        pass



for col in X_test.columns:

    label_encoder = LabelEncoder()

    try:

        X_test[col] = X_test[col].fillna("")

        X_test[col] = label_encoder.fit_transform(X_test[col])

    except Exception as e :

        print(col)

        print(e)

        pass



# print(X_train.head())





#Feautre engineering using PCA
pca = PCA(n_components=2).fit(X_train)

X_train = pca.transform(X_train)

X_test_df = X_test

pca = PCA(n_components=2).fit(X_test)

X_test = pca.transform(X_test)

print(X_train)
#Classifier
clf = RandomForestClassifier().fit(X_train, y_train)

y_test = clf.predict(X_test)

#Accuracy
train_accuracy = clf.score(X_train, y_train)

test_accuracy = clf.score(X_test, y_test)

print(train_accuracy, test_accuracy)
# AUC & #ROC
probas_ = clf.predict_proba(X_test)

# print(probas_[:, 1])

fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])

roc_auc_lr = auc(fpr, tpr)

print(roc_auc_lr)
# Save to CSV
output = X_test_df

output['Survived'] = y_test

output.to_csv('output.csv', index=False)