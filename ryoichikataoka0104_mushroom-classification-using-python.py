import numpy as np 
import pandas as pd 
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white") 
sns.set(style="whitegrid", color_codes=True)

print(os.listdir("../input"))
%matplotlib inline
df = pd.read_csv("../input/mushrooms.csv")
df.head()
df.info()
# class label encoding
from sklearn.preprocessing import LabelEncoder
# create instance from LabelEncoder
class_label = LabelEncoder()
# class label to integer
# edible = 0, poisonous = 1
df["class"] = class_label.fit_transform(df["class"].values)
df.head()
# one-hot encoding
df_dummies = pd.get_dummies(df)
df_dummies.head()
# split training and test datase
from sklearn.model_selection import train_test_split
X, y = df_dummies.iloc[:, 1:].values, df_dummies.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    stratify=y, random_state=1)
print("Training Samples: {}".format(X_train.shape[0]))
print("Test Samples: {}".format(X_test.shape[0]))
# Verify Stratification Sampling
print("Class label counts in y: ", np.bincount(y))
print("Class label counts in y_train:", np.bincount(y_train))
print("Class label counts in y_test:", np.bincount(y_test))
# 3 Types Classification
from sklearn.linear_model import LogisticRegression as LOR
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier as RFC
# For Confusion matrix
from sklearn.metrics import confusion_matrix
# For Precision
from sklearn.metrics import precision_score 
# For Recall
from sklearn.metrics import recall_score
# For F1 Score
from sklearn.metrics import f1_score
# LogisticRegression
lor = LOR(penalty='l2', random_state=1).fit(X_train, y_train)
y_train_pred_lor = lor.predict(X_train)
y_test_pred_lor = lor.predict(X_test)
y_true = y_test
# confmat_lor = confusion_matrix(y_true, y_test_pred_lor)
labels = sorted(list(set(y_true)))
confmat_lor = confusion_matrix(y_true, y_test_pred_lor, labels=labels)
df_cmx_lor = pd.DataFrame(confmat_lor, index=labels, columns=labels)
plt.figure(figsize = (10, 7))
sns.heatmap(df_cmx_lor, annot=True, fmt="d", linewidths=.5)
plt.show()
print(confmat_lor)
print("Precision: %3f" % precision_score(y_true, y_test_pred_lor))
print("Recall: %3f" % recall_score(y_true, y_test_pred_lor))
print("F1: %3f" % f1_score(y_true, y_test_pred_lor))

# SVM
svm = SVC(random_state=1, gamma=0.10).fit(X_train, y_train)
y_train_pred_svm = svm.predict(X_train)
y_test_pred_svm = svm.predict(X_test)
y_true = y_test
# confmat_svm = confusion_matrix(y_true, y_test_pred_svm)
labels = sorted(list(set(y_true)))
confmat_svm = confusion_matrix(y_true, y_test_pred_svm, labels=labels)
df_cmx_svm = pd.DataFrame(confmat_svm, index=labels, columns=labels)

plt.figure(figsize = (10, 7))
sns.heatmap(df_cmx_svm, annot=True, fmt="d", linewidths=.5)
plt.show()
print(confmat_svm)
print("Precision: %3f" % precision_score(y_true, y_test_pred_svm))
print("Recall: %3f" % recall_score(y_true, y_test_pred_svm))
print("F1: %3f" % f1_score(y_true, y_test_pred_svm))

# RandomForestRegressor
rfc = RFC(random_state=1, n_estimators=25, n_jobs=2).fit(X_train, y_train)
y_train_pred_rfc = rfc.predict(X_train)
y_test_pred_rfc = rfc.predict(X_test)
y_true = y_test

# confmat_rfc = confusion_matrix(y_true, y_test_pred_rfc)
labels = sorted(list(set(y_true)))
confmat_rfc = confusion_matrix(y_true, y_test_pred_rfc, labels=labels)
df_cmx_rfc = pd.DataFrame(confmat_rfc, index=labels, columns=labels)

plt.figure(figsize = (10, 7))
sns.heatmap(df_cmx_rfc, annot=True, fmt="d", linewidths=.5)
plt.show()
print(confmat_rfc)
print("Precision: %3f" % precision_score(y_true, y_test_pred_rfc))
print("Recall: %3f" % recall_score(y_true, y_test_pred_rfc))
print("F1: %3f" % f1_score(y_true, y_test_pred_rfc))
