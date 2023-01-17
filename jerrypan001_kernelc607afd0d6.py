import csv as csv 

import numpy as np

import pandas as pd

import matplotlib.cm as cm

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from scipy import stats  





from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold,GridSearchCV

traindf = pd.read_csv('../input/train.csv', header=0) 



x_train = traindf.drop(['id', 'species'], axis=1)

y_train = traindf.pop('species')



scaler = StandardScaler().fit(x_train) #find mean and std for the standardization

x_train = scaler.transform(x_train) #standardize the training values



kfold = KFold(n_splits=5, shuffle=True, random_state=4)

rf = ExtraTreesClassifier(n_estimators=500, random_state=0)

rf_validation=[rf.fit(x_train[train], y_train[train]).score(x_train[test], y_train[test]).mean() \

               for train, test in kfold.split(x_train)]




importances = rf.feature_importances_

indices = np.argsort(importances)[::-1]

imp_std = np.std([est.feature_importances_ for est in rf.estimators_], axis=0)



fig = plt.figure(figsize=(18, 10))

gs1 = gridspec.GridSpec(1, 2)#, height_ratios=[1, 1]) 

ax2 =fig.add_subplot(gs1[1])

ax2.margins(0.05) 



ax2.bar(range(20), importances[indices][-20:], \

       color="#000000", yerr=imp_std[indices][-20:], ecolor='#000000', align="center")

ax2.set_xticks(range(20))

ax2.set_xticklabels(indices[-20:])

ax2.set_xlim([-1, 20])

ax2.set_ylim([0, 0.035])

ax2.set_xlabel('Feature #')

 

ax2.set_ylabel(' Normalized Importance')

ax2.set_title('Last 10 Important Features')

gs1.tight_layout(fig)

#plt.show()
import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import learning_curve, GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier



np.random.seed(42)



train = pd.read_csv('../input/train.csv')

x_train = train.drop(['id', 'species', 'margin7', 'margin15', 'margin33', 'texture14','margin51','margin60'], axis=1).values

le = LabelEncoder().fit(train['species'])

y_train = le.transform(train['species'])

scaler = StandardScaler().fit(x_train)

x_train = scaler.transform(x_train)



test = pd.read_csv('../input/test.csv')

test = test.drop(['margin7', 'margin15', 'margin33', 'texture14','margin51','margin60'], axis=1)

test_ids = test.pop('id')

x_test = test.values

scaler = StandardScaler().fit(x_test)

x_test = scaler.transform(x_test)



#params = {'C':[1, 10, 50, 100, 500, 1000, 2000], 'tol': [0.001, 0.0001, 0.005], 'solver':  ["newton-cg"]}

#log_reg = LogisticRegression(multi_class="multinomial")

#clf = GridSearchCV(log_reg, params, scoring='log_loss', refit='True', n_jobs=1, cv=5)

#clf.fit(x_train, y_train)

#y_test = clf.predict_proba(x_test)









log_reg = LogisticRegression(C=2200, multi_class="multinomial", tol=0.0001, solver='newton-cg')

log_reg.fit(x_train, y_train)

y_test = log_reg.predict_proba(x_test)



#params = {'n_estimators':[1, 10, 50, 100, 500]}

#random_forest = RandomForestClassifier()

#clf = GridSearchCV(random_forest, params, scoring='log_loss', refit='True', n_jobs=1, cv=5)

#clf.fit(x_train, y_train)

#y_test = clf.predict_proba(x_test)



submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)

submission.to_csv('submission.csv')