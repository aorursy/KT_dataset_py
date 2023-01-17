# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
data.head()
data.describe()
print(np.sum(np.array(data.iloc[:,-1]) == 1) + np.sum(np.array(data.iloc[:,-1]) == 0))

data.shape
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
X = data.iloc[:,:-1]

Y = data.iloc[:,-1:]

X_tr,X_ts,Y_tr,Y_ts = train_test_split(X,Y,test_size = 0.3)

m_tr = Y_tr.shape[0]

m_ts = Y_ts.shape[0]

y_tr = np.array(Y_tr)[:,0]

y_ts = np.array(Y_ts)[:,0]

print(X.shape,Y.shape)

print(X_tr.shape,y_tr.shape)
clf = LogisticRegression(random_state=0).fit(X_tr, Y_tr)

y_pr = clf.predict(X_ts)

np.sum(y_pr==np.array(Y_ts).flatten()) / m_ts

clf.score(X_ts, y_ts)
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score,roc_curve

print(confusion_matrix(y_pr,y_ts))

print('precision_score = ',precision_score(y_pr,y_ts))

print('recall_score = ',recall_score(y_pr,y_ts))

print('f1_score = ',f1_score(y_pr,y_ts))
roc_curve(y_pr,y_ts)
from sklearn.metrics import roc_curve,roc_auc_score



import matplotlib.pyplot as plt



roc_auc_score(list(y_ts),list(y_pr))

fpr,tpr,thresholds=roc_curve(y_ts,y_pr)

plt.plot([0,1],[0,1],linestyle='--')

plt.plot(fpr,tpr,marker='.')

plt.show()