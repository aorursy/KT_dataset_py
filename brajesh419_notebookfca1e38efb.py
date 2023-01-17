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
import warnings
warnings.filterwarnings('ignore')
train=pd.read_csv('../input/mobile-price-range-prediction-is2020-v2/train_data.csv')
test=pd.read_csv('../input/mobile-price-range-prediction-is2020-v2/test_data.csv')
sample_submission= pd.read_csv('../input/mobile-price-range-prediction-is2020-v2/sample_submission.csv')
train.head()
test.head()
print(train.shape,test.shape)
x_train=train.drop(['price_range','id'],axis=1)
y_train=train['price_range']
x_test=test.drop(['id'],axis=1)
x_test.head()
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
y_pred = logisticRegr.predict(x_test)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(logisticRegr,x_train,y_train,cv=5)
print(scores)
print(scores.mean())
from sklearn import svm
clf = svm.SVC()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(logisticRegr,x_train,y_train,cv=5)
print(scores)
print(scores.mean())
data={'id':sample_submission['id'],'price_range':y_pred}
result_dtr=pd.DataFrame(data)
result_dtr.to_csv('/kaggle/working/result_svm.csv',index=False)
from sklearn.tree import DecisionTreeClassifier
Dec_tree=DecisionTreeClassifier()
Dec_tree=Dec_tree.fit(x_train,y_train)
y_pred=Dec_tree.predict(x_test)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(logisticRegr,x_train,y_train,cv=5)
print(scores)
print(scores.mean())
data={'id':sample_submission['id'],'price_range': y_pred}
result_dtr=pd.DataFrame(data)
result_dtr.to_csv('/kaggle/working/result_dtr.csv',index=False)
from sklearn.ensemble import RandomForestClassifier
Ran_forest=RandomForestClassifier()
Ran_forest=Ran_forest.fit(x_train,y_train)
y_pred=Ran_forest.predict(x_test)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(logisticRegr,x_train,y_train,cv=5)
print(scores)
print(scores.mean())
data={'id':sample_submission['id'],'price_range':y_pred}
result_rf=pd.DataFrame(data)
result_rf.to_csv('/kaggle/working/result_rf.csv',index=False)