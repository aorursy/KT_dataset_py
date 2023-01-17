#Import Module

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
#Dataset

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

Sub = pd.read_csv('../input/sample_submission.csv')



#Create copy

df = df_train.copy()

df
#TF-IDF 

Vectorize = TfidfVectorizer(stop_words='english', token_pattern=r'\w{1,}', max_features=35000)

X = Vectorize.fit_transform(df["comment_text"])

y = np.where(df_train['target'] >= 0.5, 1, 0)

test_X = Vectorize.transform(df_test["comment_text"])

#split dataset 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Logistic regression model

clf = LogisticRegression(C=32,dual=False,n_jobs=-2,solver='sag')

clf.fit(X,y)
#Prediction of Logistic regression

y_pred=clf.predict(X)

print("Accuracy is {0:.2f}%".format(accuracy_score(y,y_pred)))
#F-measure

print(classification_report(y, y_pred))
#Plot graph

sns.set_palette("winter_r", 8)

fpr, tpr, thr = roc_curve(y, clf.predict_proba(X)[:,1])

plt.figure(figsize=(10, 8))

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic Plot')

auc = auc(fpr, tpr) * 100

plt.legend(["AUC {0:.3f}".format(auc)]);
predictions = clf.predict_proba(test_X)[:,1]
Sub['prediction'] = predictions

Sub.to_csv('submission.csv', index=False)
Sub.head()