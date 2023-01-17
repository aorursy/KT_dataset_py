import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore', category=FutureWarning)



plt.style.use("seaborn-dark")

np.random.seed(42)
data = pd.read_csv('../input/santander-customer-satisfaction/train.csv').drop('ID', axis=1)

data.head()
data.shape
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.axis('equal')

labels = data['TARGET'].unique()

target = data['TARGET'].value_counts()

ax.pie(target, labels = labels,autopct='%1.2f%%')

plt.show()
X = data.loc[:,data.columns != 'TARGET']

y = data['TARGET']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(X, y,

                                                   stratify = y,

                                                   test_size = 0.10)
from xgboost import XGBClassifier

model = XGBClassifier()
X_train.shape
model = XGBClassifier(scale_pos_weight=1)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_pred_prob = model.predict_proba(X_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



cm = confusion_matrix(y_test, y_pred)

print(cm)

print("----Classification Report----")

print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve



logit_roc_auc = roc_auc_score(y_test, y_pred)

logit_roc_auc

# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)



# plt.figure()

# plt.plot(fpr, tpr, label='Classification (area = %0.2f)' % logit_roc_auc)

# plt.plot([0, 1], [0, 1],'r--')

# plt.xlim([0.0, 1.0])

# plt.ylim([0.0, 1.05])

# plt.xlabel('False Positive Rate')

# plt.ylabel('True Positive Rate')

# plt.legend(loc="lower right")

# # plt.savefig('Log_ROC')

# plt.show()
test_submit = pd.read_csv('../input/santander-customer-satisfaction/test.csv').drop('ID', axis=1)

test_submit.head()
y_pred = model.predict(test_submit)

y_pred
submit = pd.read_csv('../input/santander-customer-satisfaction/sample_submission.csv')

submit.loc[:,'TARGET'] = y_pred
submit.to_csv('submission1.csv' , index=False,header=1)



submit.head()