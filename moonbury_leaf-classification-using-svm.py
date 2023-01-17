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
from sklearn.preprocessing import LabelEncoder



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



X_train = train.drop(['species', 'id'], axis=1) 

le = LabelEncoder().fit(train['species']) 

y_train = le.transform(train['species']) 



X_test = test.drop(['id'], axis=1)
from sklearn.svm import SVC

classifier = SVC(kernel="linear",degree=9,C=0.025,probability=True)

classifier.fit(X_train, y_train)

#y_pred = classifier.predict(X_test)



y_pred = classifier.predict_proba(X_test)

#print(y_pred)
test_ids = test.pop('id') #Id column for submission file



submission = pd.DataFrame(y_pred, index=test_ids, columns=le.classes_) 

#print (submission.head(4))

submission.to_csv('submission_leaf_classification.csv') 
#check log loss

from sklearn.metrics import log_loss

#labels = LabelEncoder().fit(train['species'])

y_pred = classifier.predict_proba(X_train)

#y = labels.transform(train['species'])

log_loss(y_train, y_pred)
from sklearn.preprocessing import StandardScaler

from sklearn import linear_model

linear_regression = linear_model.LinearRegression(normalize=False, fit_intercept=True)
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn import linear_model

linear_regression = linear_model.LinearRegression(normalize=False, fit_intercept=True)

standardization = StandardScaler()

Stand_coef_linear_reg = make_pipeline(standardization, linear_regression)

linear_predictor = make_pipeline(linear_regression)
linear_regression.fit(X_train,y_train)

X_train.columns[:-1]

coef = sorted(zip(map(abs,linear_regression.coef_), X_train.columns[:-1]), reverse=True)

for i in range(10):

    print ("%6.2f %s" % (coef[i]))
Stand_coef_linear_reg.fit(X_train,y_train)

coef = sorted(zip(map(abs,Stand_coef_linear_reg.steps[1][1].coef_), X_train.columns[:-1]), reverse=True)

for i in range(10):

    print ("%6.2f %s" % (coef[i]))
predictor = 'margin6'

x_range = [X_train[predictor].min(),X_train[predictor].max()]

y_range = [y_train.min(),y_train.max()]



x = X_train[predictor].values.reshape((990,1))

xt = np.arange(0,50,0.1).reshape((50/0.1,1))

X_train['target']=y_train

scatter = X_train.plot(kind='scatter', x=predictor, y='target', xlim=x_range, ylim=y_range)

regr_line = scatter.plot(xt, linear_predictor.fit(x,y_train).predict(xt), '-', color='red', linewidth=2)