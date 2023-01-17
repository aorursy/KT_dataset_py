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
train=pd.read_csv("/kaggle/input/mobile-price-classification/train.csv")
train.head()
train.info()
train.describe()
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
plt.figure(figsize=(15,15))
style.use('classic')
sns.heatmap(train.corr(),annot=True,cmap="rainbow")
sns.countplot(train["price_range"])
sns.pairplot(train)
x=train.drop("price_range",axis=1)
y=train["price_range"]
from sklearn.model_selection import train_test_split,KFold,cross_val_score
xr,xt,yr,yt=train_test_split(x,y,test_size=0.1)
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor,XGBRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression,SGDRegressor
from sklearn.svm import SVC,SVR
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMRegressor,LGBMClassifier
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state=42) 
x1,y1= sm.fit_sample(x,y) 
model=LGBMClassifier(n_estimators=1000)
model.fit(x1,y1)
kfold=KFold(n_splits=10)
print(model)
res=cross_val_score(model,x,y,cv=kfold)
print(res.mean()*100)
yp=model.predict(xt)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(accuracy_score(yt,yp))
print(classification_report(yt,yp))
sns.heatmap(confusion_matrix(yt,yp),annot=True,cmap='summer')
test=pd.read_csv("/kaggle/input/mobile-price-classification/test.csv")
test.head()
pred=test.drop('id',axis=1)
pred.head()
yp=model.predict(pred)
y1=pd.DataFrame(yp)
test=pd.concat([test,y1],axis=1)
test.columns=[           'id', 'battery_power',          'blue',   'clock_speed',
            'dual_sim',            'fc',        'four_g',    'int_memory',
               'm_dep',     'mobile_wt',       'n_cores',            'pc',
           'px_height',      'px_width',           'ram',          'sc_h',
                'sc_w',     'talk_time',       'three_g',  'touch_screen',
                'wifi',               'price_range']
test.to_csv('sub1.csv',columns=["id","price_range"],index=False)