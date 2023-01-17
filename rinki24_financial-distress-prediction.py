import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
data=pd.read_csv('../input/dataset/Financial Distress.csv')
data.head()
data.columns
data['Financial Distress'].value_counts()
X, y = data.loc[:,data.columns!='Financial Distress'], data.loc[:,'Financial Distress'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=123,stratify=data['Financial Distress'])
# We have used stratified above to split the data distribution in equal manner
print(pd.value_counts(y_train)/y_train.size * 100)
print(pd.value_counts(y_test)/y_test.size * 100)
from sklearn.ensemble import RandomForestClassifier
# train model
rfc = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)

# predict on test set
rfc_pred = rfc.predict(X_test)

accuracy_score(y_test, rfc_pred)
# f1 score
f1_score(y_test, rfc_pred)

# confusion matrix
pd.DataFrame(confusion_matrix(y_test, rfc_pred))
# recall score
recall_score(y_test, rfc_pred)
from sklearn.utils import resample

X['class']=y
X.columns
# separate minority and majority classes
not_distress = X[X['class']==0]
distress = X[X['class']==1]

# upsample minority
fraud_upsampled = resample(distress,
                          replace=True, # sample with replacement
                          n_samples=len(not_distress), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([not_distress, fraud_upsampled])

# check new class counts
upsampled['class'].value_counts()
# trying logistic regression again with the balanced dataset
y_train = upsampled['class']
X_train = upsampled.drop('class', axis=1)

upsampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)

upsampled_pred = upsampled.predict(X_test)

# Checking accuracy
accuracy_score(y_test, upsampled_pred)
# f1 score
f1_score(y_test, upsampled_pred)
# confusion matrix
pd.DataFrame(confusion_matrix(y_test, upsampled_pred))
recall_score(y_test, upsampled_pred)
# downsample majority
not_fraud_downsampled = resample(not_distress,
                                replace = False, # sample without replacement
                                n_samples = len(distress), # match minority n
                                random_state = 27) # reproducible results

# combine minority and downsampled majority
downsampled = pd.concat([not_fraud_downsampled, distress])

# checking counts
downsampled['class'].value_counts()
y_train = downsampled['class']
X_train = downsampled.drop('class', axis=1)

undersampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)

undersampled_pred = undersampled.predict(X_test)
# Checking accuracy
accuracy_score(y_test, undersampled_pred)
# f1 score
f1_score(y_test, undersampled_pred)
# confusion matrix
pd.DataFrame(confusion_matrix(y_test, undersampled_pred))
recall_score(y_test, undersampled_pred)
from imblearn.over_sampling import SMOTE

# Separate input features and target
y = data['Financial Distress']
X = data.drop('Financial Distress', axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

sm = SMOTE(random_state=27, ratio=1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)
smote = LogisticRegression(solver='liblinear').fit(X_train, y_train)

smote_pred = smote.predict(X_test)

# Checking accuracy
accuracy_score(y_test, smote_pred)

# f1 score
f1_score(y_test, smote_pred, average='weighted')

# confustion matrix
pd.DataFrame(confusion_matrix(y_test, smote_pred))

recall_score(y_test, smote_pred)
