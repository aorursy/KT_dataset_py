# Using LogisticRegression on the cancer dataset. 
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
%matplotlib inline

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state= 42)

Log_reg = LogisticRegression()
Log_reg.fit(X_train, y_train)
print('Accuracy on the training subset: {:.3f}'.format(Log_reg.score(X_train, y_train)) )
print('Accuracy on the test subset: {:.3f}'.format(Log_reg.score(X_test, y_test)) )
# Using regularization to overcome overfitting, therefore setting C=100
Log_reg100 =LogisticRegression(C=100)
Log_reg100.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(Log_reg100.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(Log_reg100.score(X_test, y_test)))
