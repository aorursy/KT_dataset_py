# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
x=train.drop(['ID_code','target'], axis=1)

y=train.target

x_t=test.drop(['ID_code'],axis=1)
from imblearn.under_sampling import RandomUnderSampler

ros = RandomUnderSampler(random_state=0)

ros.fit(x, y)

x_resampled, y_resampled = ros.fit_resample(x, y)
np.unique(y_resampled,return_counts=True)
x.isnull().sum()
train.corr()
x_t.head()

x.describe



train_x, val_x, train_y, val_y = train_test_split(x_resampled, y_resampled, test_size = 0.25, random_state = 0)
rf1=LogisticRegression()

rf1.fit(train_x,train_y)

#y_pred_1=rf1.predict_proba(val_x)

#accuracy_score(y_pred_1, val_y)

y_pred_f=rf1.predict_proba(x_t)
#svm=SVC(C=1000,probability=True)

#svm.fit(train_x, train_y)

#y_pred_f=svm.predict(x_t)
#rf2=RandomForestClassifier(n_estimators=10)

#rf2.fit(train_x,train_y)

#y_pred_2=rf2.predict(val_x)

#accuracy_score(y_pred_2, val_y)
#rf3=RandomForestClassifier(n_estimators=10)

#rf3.fit(train_x,train_y)

#y_pred_3=rf3.predict(val_x)

#accuracy_score(y_pred_3, val_y)
#clf = MLPClassifier(solver='sgd', alpha=0.0001,hidden_layer_sizes=(200,100,100),max_iter=500, random_state=1)

#clf.fit(train_x, train_y)

#y_pred=clf.predict(val_x)

#accuracy_score(y_pred, val_y)
#clf_l = SVC(kernel='linear')    

#clf_l.fit(train_x,train_y) 
ndf= pd.DataFrame({'ID_code':test['ID_code'], 'target': y_pred_f[:,0] })

ndf.to_csv('Submission3', index=False)