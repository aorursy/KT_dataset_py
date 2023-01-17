# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
cc_data = pd.read_csv('../input/creditcard.csv')

cc_data.head()
cc_data['Amount'].describe()
# Let's check the amount for both normal and fraud transactions

cc_data[cc_data['Class'] == 0]['Amount'].describe()
cc_data[cc_data['Class'] == 1]['Amount'].describe()
cc_data['Time'].hist()
from sklearn.cross_validation import train_test_split, KFold, StratifiedShuffleSplit

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve
shuffle_index = StratifiedShuffleSplit(cc_data['Class'], 1, test_size=0.15)

for train_index, test_index in shuffle_index:

    credit_train = cc_data.iloc[train_index,:30]

    credit_test = cc_data.iloc[test_index, :30]

    status_train = cc_data.iloc[train_index,30]

    status_test = cc_data.iloc[test_index,30]
model1 = LogisticRegressionCV(scoring = 'roc_auc', class_weight='balanced')

model1.fit(credit_train, status_train)
predict_result=model1.predict(credit_test)
print(classification_report(status_test, predict_result))
print(pd.DataFrame(confusion_matrix(status_test, predict_result), index=['Actual Healthy', 'Actual Default'], 

                   columns = ['Pred. Healthy', 'Pred. Default']))

print('Area under the curve is',roc_auc_score(status_test, predict_result))
