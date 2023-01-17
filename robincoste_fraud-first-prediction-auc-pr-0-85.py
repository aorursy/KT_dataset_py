import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LinearRegression

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import auc

import pylab as pl





data = pd.read_csv('../input/creditcard.csv')
data['Night'] = ((np.mod(np.floor(data['Time']/60/60),24) <= 9)*1).astype('category')

data['Hour'] = (np.floor(data['Time']/60/60)).astype('category')

data['roundAmount'] = (np.round(data['Amount']/100)*100).astype('category')

del data['Time']

data= pd.get_dummies(data)
y = data['Class']

X = data.drop(['Class'], axis=1).values

skf = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state =43)

for valid_train_is, valid_test_is in skf.split(X, y):

    X_train = X[valid_train_is]

    y_train = y[valid_train_is]

    X_test = X[valid_test_is]

    y_test = y[valid_test_is]



    clf = ExtraTreesClassifier(n_estimators =200)

    clf.fit(X_train, y_train)

    clf.predict(X_test)

    

    probas_ = clf.predict_proba(X_test)

    precision, recall, thresholds = precision_recall_curve(y_test, probas_[:, 1])

    area = auc(recall, precision)

    print("Area Under P-R Curve: ",area)
pl.clf()

pl.plot(recall, precision, label='Precision-Recall curve')

pl.xlabel('Recall')

pl.ylabel('Precision')

pl.ylim([0.0, 1.05])

pl.xlim([0.0, 1.0])

pl.title('Precision-Recall example: AUC=%0.2f' % area)

pl.legend(loc="lower left")

pl.show()
