import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train=pd.read_csv('../input/titanic/train.csv')

train.head()
train.dtypes
train.drop(['PassengerId','Ticket','Name','Cabin','Embarked'],inplace=True,axis=1)
train.head()
train['Survived'].value_counts()
train=pd.get_dummies(train, columns =['Sex','Pclass'], drop_first=True)

train.head()
train=train.astype(np.float64)

train.dtypes
train.isnull().sum()
train['Age'].fillna(train['Age'].median(), inplace = True)
train.isnull().sum()
X=train.iloc[:,1:8]

y=train.Survived
X.columns
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=500)

logreg.fit(X_train,y_train.ravel())



y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
#confusion matrix

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)



#different perfomance matric 

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))



#ROC

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve



logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r-')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

plt.legend(loc="lower right")

plt.show()