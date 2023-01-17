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
import numpy as np

import pandas as pd

import os

credit = pd.read_csv("../input/creditcardfraud/creditcard.csv")

credit.head()
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(credit.as_matrix(), test_size = 0.2,random_state = 42)
Y_train = train_set[:,30].copy()

X_train = train_set[:,0:29]
Y_test = test_set[:,30].copy()

X_test = test_set[:,0:29]
from sklearn import preprocessing

X_train_scales = preprocessing.scale(X_train)

X_test_scales = preprocessing.scale(X_test)

Y_train_1 = (Y_train == 1)

Y_test_1 = (Y_test == 1)
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC

poly_kernal_svc_clf = SVC(kernel = "poly",degree = 3,coef0 = 1,C=5)

poly_kernal_svc_clf.fit(X_train_scales,Y_train)
y_svc_train_predict = cross_val_predict(poly_kernal_svc_clf,X_train_scales,Y_train,cv=3)

confusion_matrix(y_svc_train_predict,Y_train)
poly_kernal_svc_clf.predict(X_test_scales)

poly_kernal_svc_clf.score(X_test_scales,Y_test)
poly_kernal_svc_clf.score(X_train_scales,Y_train)
y_svc_test_predict = cross_val_predict(poly_kernal_svc_clf,X_test_scales,Y_test,cv=3)

confusion_matrix(y_svc_test_predict,Y_test)
from sklearn.metrics import precision_score,recall_score
precision_score(Y_test,y_svc_test_predict)

recall_score(Y_test,y_svc_test_predict)
precision_score(Y_test,y_svc_test_predict)