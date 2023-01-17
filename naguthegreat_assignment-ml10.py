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
train=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020/train_data.csv')
test=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020/test_data.csv')
sample_submission=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020/sample_submission.csv')
train.head()
test.head()
sample_submission.head()
print(train.shape,test.shape,sample_submission.shape)
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
train_new=train.drop(columns=['id'])
test_new=test.drop(columns=['id'])
train_new.info()
train1=train_new.drop(columns=['price_range'])
train2=train_new['price_range']
from sklearn.preprocessing import StandardScaler
train1_scale=StandardScaler().fit_transform(train1)
pd.DataFrame(train1_scale).head()
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression().fit(train1_scale,train2)
y_pred=lr.predict(test_new)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(LogisticRegression(),train1,train2,cv=5)
print(scores)
print(scores.mean())
data={'id':sample_submission['id'],'price_range':y_pred}
result=pd.DataFrame(data)
result.to_csv('/kaggle/working/result_svm.csv',index=False)
output=pd.read_csv('/kaggle/working/result_svm.csv')
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(criterion='gini',n_estimators=25,random_state=0).fit(train1,train2)
y_pred=forest.predict(test_new)
from sklearn.svm import SVC
svm=SVC(C=1.0,kernel='linear',random_state=0)
svm=svm.fit(train1,train2)
y_pred=svm.predict(test_new)
print(svm.score(train1,train2))
scores=cross_val_score(SVC(),train1,train2,cv=5)
print(scores)
print(scores.mean())
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=0)
tree=tree.fit(train1,train2)
y_pred=tree.predict(test_new)
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb=nb.fit(train1,train2)
y_pred=nb.predict(test_new)

scores=cross_val_score(GaussianNB(),train1,train2,cv=5)
print(scores)
print(scores.mean())
data={'id': sample_submission['id'],
      'price_range':y_pred}
res=pd.DataFrame(data)
res.to_csv('/kaggle/working/result_assign.csv',index=False)
output=pd.read_csv('/kaggle/working/result_assign.csv')