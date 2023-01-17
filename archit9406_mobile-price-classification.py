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


import numpy as np

import matplotlib.pyplot as plt

import pandas as pd


train = pd.read_csv('../input/mobile-price-classification/train.csv')

test = pd.read_csv('../input/mobile-price-classification/train.csv')



X_train = train.drop(["price_range"],axis=1)

y_train = train["price_range"]
print(train.columns)

print(train.head())



print(test.columns)

print(test.head())
train.info()

train.describe()
#buliding the optimal data using automatic backward elimnation

import statsmodels.api as sm

SL = 0.05

X_train_arr=X_train.values

X_opt = X_train_arr[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]

X=np.append(arr=np.ones((2000,1)).astype(int),values=X_train,axis=1)

regressor_ols=sm.OLS(endog=y_train,exog=X_opt).fit()



print(regressor_ols.summary())

li=[]

def backwardElimination(x, sl):

    numVars = len(x[0])

    for i in range(0, numVars):

        regressor_OLS = sm.OLS(y_train, x).fit()

        maxVar = max(regressor_OLS.pvalues)

        if maxVar > sl:

            for j in range(0, numVars - i):

                if (regressor_OLS.pvalues[j] == maxVar):

                    x = np.delete(x, j, 1)

                    li.append(j)

    regressor_OLS.summary()

    return x

X_Modeled = backwardElimination(X_opt, SL)

test=test.values

test=np.delete(test,5,1)

test=np.delete(test,6,1)

test=np.delete(test,12,1)
len(X_Modeled[0])
#Applying feature scaling



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_Modeled= sc.fit_transform(X_Modeled)

test = sc.fit_transform(test)
len(test[0])
from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_Modeled, y_train, test_size = 0.3, random_state = 0)
from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train1, y_train1)
# Predicting the Test set results

y_pred_SVC= classifier.predict(X_test1)
#calculating accuracy

acc_SVC= round(classifier.score(X_train1,y_train1) * 100, 2)

print(acc_SVC)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm_SVC= confusion_matrix(y_test1, y_pred_SVC)

print(cm_SVC)
import sklearn
from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(random_state=0)

classifier.fit(X_train1,y_train1)
# Predicting the Test set results

y_pred_logistic= classifier.predict(X_test1)
#calculating accuracy

acc_logistic= round(classifier.score(X_train1,y_train1) * 100, 2)

print(acc_SVC)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm_logistic= confusion_matrix(y_test1, y_pred_logistic)

print(cm_logistic)
from sklearn.neighbors import KNeighborsClassifier

classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

classifier.fit(X_train1,y_train1)
# Predicting the Test set results

y_pred_knn= classifier.predict(X_test1)
#calculating accuracy

acc_knn= round(classifier.score(X_train1,y_train1) * 100, 2)

print(acc_knn)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm_knn= confusion_matrix(y_test1, y_pred_knn)

print(cm_knn)
from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)

classifier.fit(X_train1,y_train1)
# Predicting the Test set results

y_pred_random= classifier.predict(X_test1)
#calculating accuracy

acc_random= round(classifier.score(X_train1,y_train1) * 100, 2)

print(acc_random)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm_random= confusion_matrix(y_test1, y_pred_random)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test1,y_pred_random))