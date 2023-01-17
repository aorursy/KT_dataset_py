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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("../input/bank-marketing/bank-additional-full.csv", sep=";")
data.head()
data.columns
data.shape
data.describe()
data.info()
data['job'].value_counts()
data['y'].value_counts()
data['y'].head()
data_new = data.copy()
y = data[['y']]

x = data_new.drop('y', axis=1)
x.shape, y.shape
y.value_counts()
y = pd.get_dummies(y, drop_first=True)
y.value_counts()
for i in x.columns:

    print("This column has categories \n: ", x[i].unique())
print('Jobs:\n', x['job'].value_counts())

print('Marital Status:\n', x['marital'].value_counts())

print('Education:\n', x['education'].value_counts())

print('Default:\n', x['default'].value_counts())

print('Housing:\n', x['housing'].value_counts())

print('Loan:\n', x['loan'].value_counts())

print('Contact:\n', x['contact'].value_counts())

print('month:\n', x['month'].value_counts())

print('day_of_week:\n', x['day_of_week'].value_counts())

print('poutcome:\n', x['poutcome'].value_counts())
x.isnull().sum()

y.isnull().sum()
x.select_dtypes(include="object")
import pandas_profiling

x.profile_report()
# Label encoder order is alphabetical

from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()

x['job']      = labelencoder_X.fit_transform(x['job']) 

x['marital']  = labelencoder_X.fit_transform(x['marital']) 

x['education']= labelencoder_X.fit_transform(x['education']) 

x['default']  = labelencoder_X.fit_transform(x['default']) 

x['housing']  = labelencoder_X.fit_transform(x['housing']) 

x['contact']  = labelencoder_X.fit_transform(x['contact'])

x['month']     = labelencoder_X.fit_transform(x['month'])

x['day_of_week'] = labelencoder_X.fit_transform(x['day_of_week'])
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))

sns.boxplot(x = 'duration', data = x, orient = 'v', ax = ax1)

ax1.set_xlabel('Calls', fontsize=10)

ax1.set_ylabel('Duration', fontsize=10)

ax1.set_title('Calls Distribution', fontsize=10)

ax1.tick_params(labelsize=10)



sns.distplot(x['duration'], ax = ax2)

sns.despine(ax = ax2)

ax2.set_xlabel('Duration Calls', fontsize=10)

ax2.set_ylabel('Occurence', fontsize=10)

ax2.set_title('Duration x Ocucurence', fontsize=10)

ax2.tick_params(labelsize=10)



plt.subplots_adjust(wspace=0.5)

plt.tight_layout()
print('1º Quartile: ', x['duration'].quantile(q = 0.25))

print('2º Quartile: ', x['duration'].quantile(q = 0.50))

print('3º Quartile: ', x['duration'].quantile(q = 0.75))

print('4º Quartile: ', x['duration'].quantile(q = 1.00))

print('Duration calls above: ', x['duration'].quantile(q = 0.75) + 

                      1.5*(x['duration'].quantile(q = 0.75) - x['duration'].quantile(q = 0.25)), 'are outliers')
print('Numerber of outliers: ', x[x['duration'] > 644.5]['duration'].count())

print('Number of clients: ', len(x))

#Outliers in %

print('Outliers are:', round(x[x['duration'] > 644.5]['duration'].count()*100/len(x),2), '%')
x['poutcome'].replace(['nonexistent', 'failure', 'success'], [1,2,3], inplace  = True)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 1)

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
x_train.head()
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

x_train = scale.fit_transform(x_train)

x_test = scale.fit_transform(x_test)
from sklearn import model_selection

from sklearn.neighbors import KNeighborsClassifier
neighbors = np.arange(0,25)
cv_scores = []
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

knn = KNeighborsClassifier(n_neighbors=22)

knn.fit(x_train, y_train)
knnpred = knn.predict(x_test)
print(confusion_matrix(y_test, knnpred))
print(round(accuracy_score(y_test, knnpred),2)*100)
KNNCV = (cross_val_score(knn, x_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
KNNCV
from sklearn.svm import SVC

svc= SVC(kernel = 'sigmoid')

svc.fit(x_train, y_train)

svcpred = svc.predict(x_test)

print(confusion_matrix(y_test, svcpred))

print(round(accuracy_score(y_test, svcpred),2)*100)

SVCCV = (cross_val_score(svc, x_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='gini') #criterion = entopy, gini

dt.fit(x_train, y_train)

pred = dt.predict(x_test)



print(confusion_matrix(y_test, pred))

print(round(accuracy_score(y_test, pred),2)*100)

DTREECV = (cross_val_score(dt, x_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression() 

logmodel.fit(x_train,y_train)

logpred = logmodel.predict(x_test)





print(confusion_matrix(y_test, logpred))

print(round(accuracy_score(y_test, logpred),2)*100)

LOGCV = (cross_val_score(logmodel, x_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 200)#criterion = entopy,gini

rfc.fit(x_train, y_train)

rfcpred = rfc.predict(x_test)



print(confusion_matrix(y_test, rfcpred ))

print(round(accuracy_score(y_test, rfcpred),2)*100)

RFCCV = (cross_val_score(rfc, x_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(x_train, y_train)

xgbprd = xgb.predict(x_test)



print(confusion_matrix(y_test, xgbprd ))

print(round(accuracy_score(y_test, xgbprd),2)*100)

XGB = (cross_val_score(estimator = xgb, X=x_train, y=y_train, cv = 10).mean())
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

gbkpred = gbk.predict(x_test)

print(confusion_matrix(y_test, gbkpred ))

print(round(accuracy_score(y_test, gbkpred),2)*100)

GBKCV = (cross_val_score(gbk, x_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
models = pd.DataFrame({'Models': ['Random Forest Classifier', 'Decision Tree Classifier', 'Support Vector Machine',

                           'K-Near Neighbors', 'Logistic Model', 'XGBoost', 'Gradient Boosting'],

                'Score':  [RFCCV, DTREECV, SVCCV, KNNCV, LOGCV, XGB, GBKCV]})

models.sort_values(by='Score', ascending=False)
from sklearn import metrics

fig, (ax, ax1) = plt.subplots(nrows = 2, ncols = 2, figsize = (15,5))

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
fig, ax_arr = plt.subplots(nrows = 2, ncols = 2, figsize = (20,15))



#LOGMODEL

probs = logmodel.predict_proba(x_test)

preds = probs[:,1]

fprlog, tprlog, thresholdlog = metrics.roc_curve(y_test, preds)

roc_auclog = metrics.auc(fprlog, tprlog)



ax_arr[0,0].plot(fprlog, tprlog, 'b', label = 'AUC = %0.2f' % roc_auclog)

ax_arr[0,0].plot([0, 1], [0, 1],'r--')

ax_arr[0,0].set_title('Receiver Operating Characteristic Logistic ',fontsize=20)

ax_arr[0,0].set_ylabel('True Positive Rate',fontsize=20)

ax_arr[0,0].set_xlabel('False Positive Rate',fontsize=15)

ax_arr[0,0].legend(loc = 'lower right', prop={'size': 16})



#RANDOM FOREST --------------------

probs = rfc.predict_proba(x_test)

preds = probs[:,1]

fprrfc, tprrfc, thresholdrfc = metrics.roc_curve(y_test, preds)

roc_aucrfc = metrics.auc(fprrfc, tprrfc)



ax_arr[0,1].plot(fprrfc, tprrfc, 'b', label = 'AUC = %0.2f' % roc_aucrfc)

ax_arr[0,1].plot([0, 1], [0, 1],'r--')

ax_arr[0,1].set_title('Receiver Operating Characteristic Random Forest ',fontsize=20)

ax_arr[0,1].set_ylabel('True Positive Rate',fontsize=20)

ax_arr[0,1].set_xlabel('False Positive Rate',fontsize=15)

ax_arr[0,1].legend(loc = 'lower right', prop={'size': 16})



#KNN----------------------

probs = knn.predict_proba(x_test)

preds = probs[:,1]

fprknn, tprknn, thresholdknn = metrics.roc_curve(y_test, preds)

roc_aucknn = metrics.auc(fprknn, tprknn)



ax_arr[1,0].plot(fprknn, tprknn, 'b', label = 'AUC = %0.2f' % roc_aucknn)

ax_arr[1,0].plot([0, 1], [0, 1],'r--')

ax_arr[1,0].set_title('Receiver Operating Characteristic KNN ',fontsize=20)

ax_arr[1,0].set_ylabel('True Positive Rate',fontsize=20)

ax_arr[1,0].set_xlabel('False Positive Rate',fontsize=15)

ax_arr[1,0].legend(loc = 'lower right', prop={'size': 16})



#DECISION TREE ---------------------

probs = dt.predict_proba(x_test)

preds = probs[:,1]

fprdtree, tprdtree, thresholddtree = metrics.roc_curve(y_test, preds)

roc_aucdtree = metrics.auc(fprdtree, tprdtree)



ax_arr[1,1].plot(fprdtree, tprdtree, 'b', label = 'AUC = %0.2f' % roc_aucdtree)

ax_arr[1,1].plot([0, 1], [0, 1],'r--')

ax_arr[1,1].set_title('Receiver Operating Characteristic Decision Tree ',fontsize=20)

ax_arr[1,1].set_ylabel('True Positive Rate',fontsize=20)

ax_arr[1,1].set_xlabel('False Positive Rate',fontsize=15)

ax_arr[1,1].legend(loc = 'lower right', prop={'size': 16})