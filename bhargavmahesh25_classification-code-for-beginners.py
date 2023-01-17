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
import matplotlib.pyplot  as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("../input/knn-project-data/KNN_Project_Data")

data.head()
data.shape
data.columns
data.describe()
data_new = data.copy()

y = data[['TARGET CLASS']]

x = data.drop('TARGET CLASS', axis=1)
x.shape
y.shape
data.info()
data.describe()
data.isnull().sum()
data.skew().plot()
data.skew()
sns.pairplot(data, hue='TARGET CLASS', palette='viridis')
data['TARGET CLASS'].value_counts()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score

k_fold = KFold(n_splits=3, shuffle=True, random_state=0)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression() 

logmodel.fit(x_train,y_train)

logpred = logmodel.predict(x_test)





print(confusion_matrix(y_test, logpred))

print(round(accuracy_score(y_test, logpred),2)*100)

LOGCV = (cross_val_score(logmodel, x_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
from sklearn import model_selection

from sklearn.neighbors import KNeighborsClassifier



#Neighbors

neighbors = np.arange(0,10)



#Create empty list that will hold cv scores

cv_scores = []



#Perform 10-fold cross validation on training set for odd values of k:

for k in neighbors:

    k_value = k+1

    knn = KNeighborsClassifier(n_neighbors = k_value, weights='uniform', p=2, metric='euclidean')

    kfold = model_selection.KFold(n_splits=10, random_state=123)

    scores = model_selection.cross_val_score(knn, x_train, y_train, cv=kfold, scoring='accuracy')

    cv_scores.append(scores.mean()*100)

    print("k=%d %0.2f (+/- %0.2f)" % (k_value, scores.mean()*100, scores.std()*100))

optimal_k = neighbors[cv_scores.index(max(cv_scores))]

print ("The optimal number of neighbors is %d with %0.1f%%" % (optimal_k, cv_scores[optimal_k]))



plt.plot(neighbors, cv_scores)

plt.xlabel('Number of Neighbors K')

plt.ylabel('Train Accuracy')

plt.show()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(x_train, y_train)

knnpred = knn.predict(x_test)



print(confusion_matrix(y_test, knnpred))

print(round(accuracy_score(y_test, knnpred),2)*100)

KNNCV = (cross_val_score(knn, x_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
from sklearn.svm import SVC

svc= SVC(kernel = 'sigmoid')

svc.fit(x_train, y_train)

svcpred = svc.predict(x_test)

print(confusion_matrix(y_test, svcpred))

print(round(accuracy_score(y_test, svcpred),2)*100)

SVCCV = (cross_val_score(svc, x_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='gini') #criterion = entopy, gini

dtree.fit(x_train, y_train)

dtreepred = dtree.predict(x_test)



print(confusion_matrix(y_test, dtreepred))

print(round(accuracy_score(y_test, dtreepred),2)*100)

DTREECV = (cross_val_score(dtree, x_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 200)#criterion = entopy,gini

rfc.fit(x_train, y_train)

rfcpred = rfc.predict(x_test)



print(confusion_matrix(y_test, rfcpred ))

print(round(accuracy_score(y_test, rfcpred),2)*100)

RFCCV = (cross_val_score(rfc, x_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
from sklearn.naive_bayes import GaussianNB

gaussiannb= GaussianNB()

gaussiannb.fit(x_train, y_train)

gaussiannbpred = gaussiannb.predict(x_test)

probs = gaussiannb.predict(x_test)



print(confusion_matrix(y_test, gaussiannbpred ))

print(round(accuracy_score(y_test, gaussiannbpred),2)*100)

GAUSIAN = (cross_val_score(gaussiannb, x_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(x_train, y_train)

xgbprd = xgb.predict(x_test)



print(confusion_matrix(y_test, xgbprd ))

print(round(accuracy_score(y_test, xgbprd),2)*100)

XGB = (cross_val_score(estimator = xgb, X = x_train, y = y_train, cv = 10).mean())


from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

gbkpred = gbk.predict(x_test)

print(confusion_matrix(y_test, gbkpred ))

print(round(accuracy_score(y_test, gbkpred),2)*100)

GBKCV = (cross_val_score(gbk, x_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
models = pd.DataFrame({'Models': ['Random Forest Classifier', 'Decision Tree Classifier', 'Support Vector Machine',

                           'K-Near Neighbors', 'Logistic Model', 'Gausian NB', 'XGBoost', 'Gradient Boosting'],

                'Score':  [RFCCV, DTREECV, SVCCV, KNNCV, LOGCV, GAUSIAN, XGB, GBKCV]})



models.sort_values(by='Score', ascending=False)
# XGBOOST ROC/ AUC , BEST MODEL

from sklearn import metrics

fig, (ax, ax1) = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))

probs = xgb.predict_proba(x_test)

preds = probs[:,1]

fprxgb, tprxgb, thresholdxgb = metrics.roc_curve(y_test, preds)

roc_aucxgb = metrics.auc(fprxgb, tprxgb)



ax.plot(fprxgb, tprxgb, 'b', label = 'AUC = %0.2f' % roc_aucxgb)

ax.plot([0, 1], [0, 1],'r--')

ax.set_title('Receiver Operating Characteristic XGBOOST ',fontsize=10)

ax.set_ylabel('True Positive Rate',fontsize=20)

ax.set_xlabel('False Positive Rate',fontsize=15)

ax.legend(loc = 'lower right', prop={'size': 16})



#Gradient

probs = gbk.predict_proba(x_test)

preds = probs[:,1]

fprgbk, tprgbk, thresholdgbk = metrics.roc_curve(y_test, preds)

roc_aucgbk = metrics.auc(fprgbk, tprgbk)



ax1.plot(fprgbk, tprgbk, 'b', label = 'AUC = %0.2f' % roc_aucgbk)

ax1.plot([0, 1], [0, 1],'r--')

ax1.set_title('Receiver Operating Characteristic GRADIENT BOOST ',fontsize=10)

ax1.set_ylabel('True Positive Rate',fontsize=20)

ax1.set_xlabel('False Positive Rate',fontsize=15)

ax1.legend(loc = 'lower right', prop={'size': 16})



plt.subplots_adjust(wspace=1)
from sklearn.metrics import classification_report

print('KNN Reports\n',classification_report(y_test, knnpred))