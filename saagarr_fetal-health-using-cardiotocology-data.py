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
data = pd.read_csv('../input/fetal-health-classification/fetal_health.csv')
data.head()
data.shape
data.describe()
data.isnull().sum()
data_corr = data.corr()
data_corr
import matplotlib.pyplot as plt
plt.figure(figsize=(18,8))
import seaborn as sns
sns.heatmap(data_corr , annot=True ,cmap ='coolwarm' ,fmt = ' .1g' ,vmin = -1 ,vmax = 1 ,center = 0 )
plt.figure(figsize=(15,6))
matrix = np.triu(data.corr())
sns.heatmap(data.corr(), annot=True, mask=matrix)
plt.figure(figsize=(15,6))
matrix = np.tril(data.corr())
sns.heatmap(data.corr(), annot=True, mask=matrix)

X = data[['baseline value','accelerations' ,'fetal_movement','uterine_contractions','severe_decelerations','prolongued_decelerations']]
y = data['fetal_health']
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),   #Step1 - normalize data
    ('clf', LogisticRegression())       #Step2 - classifier
])
pipeline.steps
from sklearn.model_selection import train_test_split
X_train ,X_test ,y_train ,y_test = train_test_split(X,y,test_size = 0.2 ,random_state=101)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.model_selection import cross_validate

scores = cross_validate(pipeline, X_train, y_train)
scores
scores['test_score'].mean()

from sklearn.neighbors import KNeighborsClassifier
k = range(1,20)
error_rate = []

for i in k:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    predictions=knn.predict(X_test)
    error_rate.append(np.mean(predictions != y_test))
plt.figure(figsize=(10,6))
plt.plot(k,error_rate,color='b',ls='--',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K')
plt.xticks(k)
plt.xlabel('K Value')
plt.ylabel('Error Rate')

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

clfs = []
clfs.append(LogisticRegression())
clfs.append(SVC())
clfs.append(KNeighborsClassifier(n_neighbors=3))
clfs.append(DecisionTreeClassifier())
clfs.append(RandomForestClassifier())
clfs.append(GradientBoostingClassifier())

for classifier in clfs:
    pipeline.set_params(clf = classifier)
    scores = cross_validate(pipeline, X_train, y_train)
    print('---------------------------------')
    print(str(classifier))
    print('-----------------------------------')
    for k, v in scores.items():
            print(k,' mean ', v.mean())
            print(k,' std ', v.std())
from xgboost import XGBClassifier
xgb_clf=XGBClassifier(learning_rate=0.005)
xgb_clf.fit(X_train,y_train)
y_pred=xgb_clf.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
from sklearn.model_selection import GridSearchCV
p_test = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,250,500,750,1000,1250,1500,1750]}

tuning = GridSearchCV(estimator = GradientBoostingClassifier(max_depth=4, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10), 
            param_grid = p_test, scoring='accuracy',n_jobs= -1, cv=5)
tuning.fit(X_train,y_train)

tuning.cv_results_, tuning.best_params_, tuning.best_score_
p_test2 = {'max_depth':[2,3,4,5,6,7] }
tuning = GridSearchCV(estimator =GradientBoostingClassifier(learning_rate=0.01,n_estimators=1250, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10), 
            param_grid = p_test2, scoring='accuracy',n_jobs=-1, cv=5)
tuning.fit(X_train,y_train)
tuning.cv_results_, tuning.best_params_, tuning.best_score_
p_test3 = {'min_samples_split':[2,4,6,8,10,20,40,60,100], 'min_samples_leaf':[1,3,5,7,9]}

tuning = GridSearchCV(estimator =GradientBoostingClassifier(learning_rate=0.01, n_estimators=1250,max_depth=4, subsample=1,max_features='sqrt', random_state=10), 
            param_grid = p_test3, scoring='accuracy',n_jobs=4, cv=5)
tuning.fit(X_train,y_train)
tuning.cv_results_, tuning.best_params_, tuning.best_score_
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
gbc_tuned = GradientBoostingClassifier(learning_rate = 0.01,max_features ='sqrt', max_depth = 4 ,n_estimators = 1250 , min_samples_split = 60 , min_samples_leaf = 1 )
gbc_tuned.fit(X_train,y_train)
y_pred = gbc_tuned.predict(X_test)
print(accuracy_score(y_test,y_pred))