import numpy as np 

import pandas as pd
data = pd.read_csv('../input/creditcard.csv')
data.head()
data.info()
data.describe()
from sklearn.preprocessing import StandardScaler
data['Vamt'] = StandardScaler().fit_transform(X=data['Amount'].values.reshape(-1, 1))
data.drop(['Time', 'Amount'], axis=1, inplace=True)



data.head()
import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline

sns.set_style('whitegrid')
plt.figure(figsize=(20,16))

sns.heatmap(data.corr())
sns.countplot(x='Class', data=data)
from collections import Counter

from sklearn.model_selection import train_test_split
X = data.drop(['Class'], axis=1)

y = data['Class']
Counter(y)
from imblearn.under_sampling.prototype_selection import RandomUnderSampler
rus = RandomUnderSampler(random_state=101)
X_res, y_res = rus.fit_sample(X, y)
Counter(y_res)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=101)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve
lr.score(X_test, y_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr,tpr)

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('ROC curve for fraud classifier')