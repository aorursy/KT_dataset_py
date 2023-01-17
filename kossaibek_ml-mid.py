import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
sns.countplot(dataset.target)
dataset.head(5)
dataset.describe()
plt.figure(figsize=(10, 7))

sns.heatmap(dataset.corr())
dataset = dataset.drop(['ID_code'],axis=1)
dataset[dataset.columns[:1]]
x = dataset[dataset.columns[1:]]

y = dataset[dataset.columns[:1]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)

x_test = sc_X.fit_transform(x_test)
from sklearn.linear_model import LogisticRegression
log_reg_cls = LogisticRegression()
log_reg_cls.fit(x_train, y_train)
y_preds_log_reg = log_reg_cls.predict(x_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print(classification_report(y_test, y_preds_log_reg))
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_preds_log_reg)
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_res, y_res = sm.fit_resample(x,y)
x_res_train, x_res_test, y_res_train, y_res_test = train_test_split(X_res, y_res, test_size = 0.2, random_state = 0)
log_reg_cls.fit(x_res_train, y_res_train)
y_res_preds_log_reg = log_reg_cls.predict(x_res_test)
print(classification_report(y_res_test, y_res_preds_log_reg))
roc_auc_score(y_res_test, y_res_preds_log_reg)
import pickle 
filename = 'LogicReg_res.sav'

pickle.dump(log_reg_cls, open(filename, 'wb'))
test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
test.head()
test_x = test[test.columns[1:]]
test[test.columns[:1]]
y_f = log_reg_cls.predict(test_x)
y_f.shape
test.ID_code.shape
my_submission = pd.DataFrame({'ID_code': test.ID_code,'target': y_f})

my_submission.to_csv('submission.csv', index=False)