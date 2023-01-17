# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import train_test_split
df= pd.read_csv('/kaggle/input/paysim1/PS_20174392719_1491204439457_log.csv')



#df['balanceOrigDelta'] = df['newbalanceOrig']-df['oldbalanceOrg']

#df['balanceDestDelta'] = df['newbalanceDest']-df['oldbalanceDest']



df.head(10)
df['deltaOrg'] = df['newbalanceOrig']-df['oldbalanceOrg']

df['deltaDest'] = df['newbalanceDest'] - df['oldbalanceDest']



df.head()
len(df[df['nameDest'].map(lambda x: x[0] == 'M')][df['isFraud'] == 1])

fraud_types = df[df['isFraud'] == 1]['type'].unique()



data = df.copy()



data = data[data['type'].isin(fraud_types)]



print(fraud_types)

data.head()
len(data[data['deltaDest'] > 0][data['isFraud'] == 1])
num_frauds = len(data[data['isFraud'] == 1])



for i in data['type'].unique():

    p = len(data[data['type'] == i][data['isFraud'] == 1])/num_frauds

    print("{:%} of fraudulent transactions are {}".format(p, i))
a = np.zeros(len(data))



for i in range(len(data)-1):

    if data['type'].iloc[i] == 'TRANSFER' and data['type'].iloc[i+1] == 'CASH_OUT':

        if data['amount'].iloc[i] == data['amount'].iloc[i+1]:

            a[i] = 1

            a[i+1] = 1
data['TransferThenCashOut'] = a





print("""{:%} of all fraudulent transactions were Transfer 

and then Cash Out for the same amount

""".format(len(data[data['isFraud'] == 1][data['TransferThenCashOut'] == 1])/num_frauds))
x = data.drop(['isFraud','nameOrig',

               'nameDest','newbalanceDest', 'isFlaggedFraud'], axis = 1)

y = data['isFraud']
x = pd.get_dummies(x)

x.head()
#Normalize

from sklearn.preprocessing import normalize



x = normalize(x)

x
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()

x_resampled, y_resampled = rus.fit_sample(x, y)

print(len(y_resampled), len(y))
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, train_size = 0.8)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier



classifier = MLPClassifier()

classifier.fit(x_train, y_train)
from sklearn.metrics import *



y_pred = classifier.predict(x_test)



print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))

print('Precision: {}'.format(precision_score(y_test, y_pred)))

print('Recall: {}'.format(recall_score(y_test, y_pred)))

print('ROC_AUC: {}'.format(roc_auc_score(y_test, y_pred)))

print('Confusion matrix:\n {}'.format(confusion_matrix(y_test, y_pred)))
y_full_pred = classifier.predict(x)



print('Accuracy: {}'.format(accuracy_score(y, y_full_pred)))

print('Precision: {}'.format(precision_score(y, y_full_pred)))

print('Recall: {}'.format(recall_score(y, y_full_pred)))

print('ROC_AUC: {}'.format(roc_auc_score(y, y_full_pred)))

print('Confusion matrix:\n {}'.format(confusion_matrix(y, y_full_pred)))