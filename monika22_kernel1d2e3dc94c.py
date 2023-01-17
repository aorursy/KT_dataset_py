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
train_data=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020/train_data.csv')
test_data=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020/test_data.csv')
sample_submission=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020/sample_submission.csv')
train_data.head()
test_data.head()
sample_submission.head()
X_train=train_data.drop(columns=['price_range','id'])
Y_train=train_data['price_range']
test_data=test_data.drop(columns=['id'])
#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(C=100,random_state=0)
lr=lr.fit(X_train,Y_train)
y_pred=lr.predict(test_data)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(LogisticRegression(),X_train,Y_train,cv=5)
print(scores)
print(scores.mean())
data={'id':sample_submission['id'],'price_range':y_pred}
result=pd.DataFrame(data)
result.to_csv('/kaggle/working/result_svm.csv',index=False)
output=pd.read_csv('/kaggle/working/result_svm.csv')
#RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(criterion='gini',n_estimators=25,random_state=0).fit(X_train,Y_train)
y_pred=forest.predict(test_data)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(RandomForestClassifier(),X_train,Y_train,cv=5)
print(scores)
print(scores.mean())

#SVM
from sklearn.svm import SVC
svm=SVC(C=1.0,kernel='linear',random_state=0)
svm=svm.fit(X_train,Y_train)
#y_pred=svm.predict(test_data)
y_pred=svm.predict(test_data)
print(svm.score(X_train,Y_train))

from sklearn.model_selection import cross_val_score
scores=cross_val_score(SVC(),X_train,Y_train,cv=5)
print(scores)
print(scores.mean())
#DECISION TREE
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=0)
tree=tree.fit(X_train,Y_train)
y_pred=tree.predict(test_data)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(DecisionTreeClassifier(),X_train,Y_train,cv=5)
print(scores)
print(scores.mean())
#NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb=nb.fit(X_train,Y_train)
y_pred=nb.predict(test_data)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(GaussianNB(),X_train,Y_train,cv=5)
print(scores)
print(scores.mean())
