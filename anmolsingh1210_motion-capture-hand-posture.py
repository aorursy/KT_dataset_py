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

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d
import matplotlib.tri as tri
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
df = pd.read_csv('../input/hand-posture/Postures.csv')
df.head()
df.tail()
print('Number of Instances  : ', df.shape[0])
print('Number of Attributes : ', df.shape[1])
df.columns
df['User'].nunique()
df.info()
df.replace({'?':np.nan},inplace=True)
df.iloc[:,2:38] = df.iloc[:,2:38].astype('float')
df.describe().T
df.isnull().sum()
df.isnull().sum()/df['X0'].count()
class1 = df[df['Class']==1]
class1.shape
class1.isnull().sum()
class2 = df[df['Class']==2]
class2.shape
class2.isnull().sum()
class3 = df[df['Class']==3]
class3.shape
class3.isnull().sum()
class4 = df[df['Class']==4]
class4.shape
class4.isnull().sum()
class5 = df[df['Class']==5]
class5.shape
class5.isnull().sum()
for i in range(2,38):
  class1.iloc[:,i] = class1.iloc[:,i].fillna(np.mean(class1.iloc[:,i]))
for i in range(2,38):
  class2.iloc[:,i] = class2.iloc[:,i].fillna(np.mean(class2.iloc[:,i]))
for i in range(2,38):
  class3.iloc[:,i] = class3.iloc[:,i].fillna(np.mean(class3.iloc[:,i]))
for i in range(2,38):
  class4.iloc[:,i] = class4.iloc[:,i].fillna(np.mean(class4.iloc[:,i]))
for i in range(2,38):
  class5.iloc[:,i] = class5.iloc[:,i].fillna(np.mean(class5.iloc[:,i]))
class1.isnull().sum()
class2.isnull().sum()
class3.isnull().sum()
class4.isnull().sum()
class5.isnull().sum()
df1 = pd.concat([class1,class2,class3,class4,class5])
df1.head()
df1.isnull().sum()
sns.set_style("whitegrid")
plt.subplots(figsize=(16, 12))
sns.boxplot(data=df1, orient='h', palette='pastel')
plt.show()
df1.drop(['X9','Y9','Z9','X10','Y10','Z10','X11','Y11','Z11'],axis=1,inplace=True)
df1.head()
df1.shape
df1['Class'] = df1['Class'].astype('category')
df1['Class'].value_counts()
plt.figure(figsize=(8,6))
plt.pie(df1['Class'].value_counts(),labels=df1['Class'].unique(),autopct='%1.2f%%')
plt.show()
dfc1 = df1[df1['Class'] == 1]
dfc1.hist(figsize=(20,20))
plt.show()
dfc2 = df1[df1['Class'] == 2]
dfc2.hist(figsize=(20,20))
plt.show()
dfc3 = df1[df1['Class'] == 3]
dfc3.hist(figsize=(20,20))
plt.show()
dfc4 = df1[df1['Class'] == 4]
dfc4.hist(figsize=(20,20))
plt.show()
dfc5 = df1[df1['Class'] == 5]
dfc5.hist(figsize=(20,20))
plt.show()
X = df1.drop(['Class','User'],axis=1)
y = df1['Class']
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 44)
%%time

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(X_train,y_train)

y_pred = dt.predict(X_test) 
print("Train Accuracy :", dt.score(X_train, y_train)*100)
print("Test Accuracy :", dt.score(X_test, y_test)*100)
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
n_folds = 10
parameters = {'criterion':['entropy','gini'],
              'max_depth':range(4,40,4),
              'min_samples_leaf':range(50,150,50),
              'min_samples_split':range(50,150,50)}
%%time

dtree = DecisionTreeClassifier(random_state=44)

RSCV = RandomizedSearchCV(estimator=dtree, param_distributions=parameters, cv=n_folds)

RSCV.fit(X_train,y_train)

print(RSCV.best_estimator_)
%%time

dtree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=12,
                              min_samples_leaf=50,
                              min_samples_split=50,
                              random_state=44)

dtree.fit(X_train,y_train)

y_pred = dtree.predict(X_test)
print("Train Accuracy :", dtree.score(X_train, y_train)*100)
print("Test Accuracy :", dtree.score(X_test, y_test)*100)
cm = confusion_matrix(y_test, y_pred)
cm = pd.DataFrame(data=cm,
                  columns=['Predicted:1','Predicted:2','Predicted:3','Predicted:4','Predicted:5'],
                  index=['Actual:1','Actual:2','Actual:3','Actual:4','Actual:5'])
plt.figure(figsize = (10,6))
sns.heatmap(cm, annot=True,fmt='d',cmap="GnBu")
plt.show()
print(classification_report(y_test,y_pred))
%%time
c = cross_val_score(dtree,X_train,y_train)
print('Variance error :',np.std(c))
print('Bias error :',1-np.mean(c))
%%time

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Train Accuracy :", knn.score(X_train, y_train)*100)
print("Test Accuracy :", knn.score(X_test, y_test)*100)
%%time

params = {'n_neighbors':range(1,5),
         'p':[1,2],
         'weights':['uniform','distance'],
         'algorithm':['ball_tree','kd_tree','brute'],
         'leaf_size':range(10,50,10)}

knn = KNeighborsClassifier()

RSCV = RandomizedSearchCV(knn, params, cv=n_folds)

RSCV.fit(X_train, y_train)

print(RSCV.best_params_)
%%time

knn = KNeighborsClassifier(weights= 'distance',
                           p= 1,
                           n_neighbors= 4,
                           leaf_size= 30,
                           algorithm= 'brute')

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
print("Train Accuracy :", knn.score(X_train, y_train)*100)
print("Test Accuracy :", knn.score(X_test, y_test)*100)
cm = confusion_matrix(y_test, y_pred)
cm = pd.DataFrame(data=cm,
                  columns=['Predicted:1','Predicted:2','Predicted:3','Predicted:4','Predicted:5'],
                  index=['Actual:1','Actual:2','Actual:3','Actual:4','Actual:5'])
plt.figure(figsize = (10,6))
sns.heatmap(cm, annot=True,fmt='d',cmap="GnBu")
plt.show()
print(classification_report(y_test,y_pred))
%%time
c = cross_val_score(knn,X_train,y_train)
print('Variance error :',np.std(c))
print('Bias error :',1-np.mean(c))
%%time

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=44)

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
print("Train Accuracy :", rfc.score(X_train, y_train)*100)
print("Test Accuracy :", rfc.score(X_test, y_test)*100)
%%time

param_dist = {'max_depth': [2, 3, 4],
              'bootstrap': [True, False],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'criterion': ['gini', 'entropy']}

cv_rf = RandomizedSearchCV(rfc, cv = 10,param_distributions=param_dist, n_jobs = 3)

cv_rf.fit(X_train,y_train)

cv_rf.best_params_
%%time

rfc = RandomForestClassifier(bootstrap=False,
                            criterion='gini',
                            max_depth=4,
                            max_features='auto',
                            random_state=44)

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
print("Train Accuracy :", rfc.score(X_train, y_train)*100)
print("Test Accuracy :", rfc.score(X_test, y_test)*100)
cm = confusion_matrix(y_test, y_pred)
cm = pd.DataFrame(data=cm,
                  columns=['Predicted:1','Predicted:2','Predicted:3','Predicted:4','Predicted:5'],
                  index=['Actual:1','Actual:2','Actual:3','Actual:4','Actual:5'])
plt.figure(figsize = (10,6))
sns.heatmap(cm, annot=True,fmt='d',cmap="GnBu")
plt.show()
print(classification_report(y_test,y_pred))
%%time
c = cross_val_score(rfc,X_train,y_train)
print('Variance error :',np.std(c))
print('Bias error :',1-np.mean(c))
%%time

from xgboost.sklearn import XGBClassifier

xgbc = XGBClassifier(random_state=44)

xgbc.fit(X_train, y_train)

y_pred = xgbc.predict(X_test)
print("Train Accuracy :", xgbc.score(X_train, y_train)*100)
print("Test Accuracy :", xgbc.score(X_test, y_test)*100)
%%time

param_dist = {'n_estimators':range(50,150,50),
              'max_depth':range(1,5)}

xgbc = XGBClassifier(random_state=44)

cv_rf = RandomizedSearchCV(xgbc, cv = 10,param_distributions=param_dist, n_jobs =-1)

cv_rf.fit(X_train,y_train)

cv_rf.best_params_
%%time

xgbc = XGBClassifier(n_estimators=100,
                             max_depth=4,
                            random_state=44)

xgbc.fit(X_train,y_train)

y_pred = xgbc.predict(X_test)
print("Train Accuracy :", xgbc.score(X_train, y_train)*100)
print("Test Accuracy :", xgbc.score(X_test, y_test)*100)
cm = confusion_matrix(y_test, y_pred)
cm = pd.DataFrame(data=cm,
                  columns=['Predicted:1','Predicted:2','Predicted:3','Predicted:4','Predicted:5'],
                  index=['Actual:1','Actual:2','Actual:3','Actual:4','Actual:5'])
plt.figure(figsize = (10,6))
sns.heatmap(cm, annot=True,fmt='d',cmap="GnBu")
plt.show()
print(classification_report(y_test,y_pred))
%%time
c = cross_val_score(xgbc,X_train,y_train)
print('Variance error :',np.std(c))
print('Bias error :',1-np.mean(c))
