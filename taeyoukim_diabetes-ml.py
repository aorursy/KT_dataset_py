import numpy as np

import pandas as pd

import matplotlib.pyplot as plt ####

%matplotlib inline
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
diabetes_data = pd.read_csv('../input/diabetes.csv')
diabetes_data.columns
diabetes_data.head()
print(diabetes_data['Outcome'].value_counts())
diabetes_data.info()
X = diabetes_data.iloc[:,:-1]

y = diabetes_data.iloc[:,-1]

y
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=156,stratify=y)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)

y_train.head()
##Logistic Regression

lr_clf=LogisticRegression()
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
def get_clf_eval(y_test, pred):

    confusion = confusion_matrix(y_test, pred)

    accuracy = accuracy_score(y_test, pred)

    precision = precision_score(y_test, pred)

    recall = recall_score(y_test, pred)

    #F1 Score add

    f1 = f1_score(y_test, pred)

    print('오차행렬')

    print(confusion)

    #f1 score print 추가

    print('정확도:{0:.4f}, 정밀도:{1:.4f}, 재현율:{2:.4f}, F1:{3:.4f}'.format(accuracy, precision, recall, f1))
get_clf_eval(y_test, pred)
pred_df = pd.DataFrame(pred)
pred_df.columns = ['Outcome']
pred_df.to_csv('Outcome.csv')