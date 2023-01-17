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
df=pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.head(20)
df.shape
df.info()
df.describe().T
df.Class.value_counts()
import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x='Class',y='Amount',data=df)
print(df.groupby('Class').aggregate({'Amount':'min'}))
print('\n')
print(df.groupby('Class').aggregate({'Amount':'max'}))
print('\n')
print(df.groupby('Class').aggregate({'Amount':'mean'}))
df.groupby('Class').aggregate({'Time':'mean'})
df.corr()
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

x=df.drop(['Class'],axis=1)
y=df['Class']
x_train,x_test,y_train,y_test=train_test_split(x,
                                              y,
                                              test_size=0.20,
                                              random_state=42)
print('x_train_shape:',x_train.shape)
print('x_test_shape:',x_test.shape)
print('y_train_shape:',y_train.shape)
print('y_test_shape:',y_test.shape)
cart=DecisionTreeClassifier()
cart_model=cart.fit(x_train,y_train)
y_pred=cart_model.predict(x_test)
y_pred_2=cart_model.predict(x_train)
print(accuracy_score(y_train,y_pred_2))
print(accuracy_score(y_test,y_pred))
from sklearn import tree
plt.figure(figsize=(30,7))
tree.plot_tree(cart_model)
plt.show()
!pip install skompiler
!pip install astor
from skompiler import skompile
print(skompile(cart_model.predict).to('python/code'))
df.head()
print('x_train_shape:',x_train.shape)
print('x_test_shape:',x_test.shape)
print('y_train_shape:',y_train.shape)
print('y_test_shape:',y_test.shape)
from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier().fit(x_train,y_train)
y_pred=rf_model.predict(x_test)
print(accuracy_score(y_test,y_pred))
import matplotlib.pyplot as plt
Importance=pd.DataFrame({'Importance':rf_model.feature_importances_*100},
            index=x_train.columns)
Importance
Importance.sort_values(by='Importance',
                      axis=0,
                      ascending=True).plot(kind='barh',color='r')
from sklearn.ensemble import GradientBoostingClassifier
gbm_model=GradientBoostingClassifier().fit(x_train,y_train)
y_pred=gbm_model.predict(x_test)
print(accuracy_score(y_test,y_pred))
Importance=pd.DataFrame({'Importance':gbm_model.feature_importances_*100},
            index=x_train.columns)
Importance
Importance.sort_values(by='Importance',
                      axis=0,
                      ascending=True).plot(kind='barh',color='b')
plt.show()
df.head(5)
print('x_train_shape:',x_train.shape)
print('x_test_shape:',x_test.shape)
print('y_train_shape:',y_train.shape)
print('y_test_shape:',y_test.shape)
from xgboost import XGBClassifier
xgb_model=XGBClassifier().fit(x_train,y_train)
y_pred=xgb_model.predict(x_test)
print(accuracy_score(y_test,y_pred))
Importance=pd.DataFrame({'Importance':xgb_model.feature_importances_*100},
            index=x_train.columns)
Importance
Importance.sort_values(by='Importance',
                      axis=0,
                      ascending=True).plot(kind='barh',color='b')
plt.show()
from lightgbm import LGBMClassifier
lgbm_model=LGBMClassifier().fit(x_train,y_train)
y_pred=lgbm_model.predict(x_test)
print(accuracy_score(y_test,y_pred))
Importance=pd.DataFrame({'Importance':lgbm_model.feature_importances_},
            index=x_train.columns)
Importance
Importance.sort_values(by='Importance',
                      axis=0,
                      ascending=True).plot(kind='barh',color='b')
plt.show()
from catboost import CatBoostClassifier
cat_model = CatBoostClassifier().fit(x_train, y_train)

y_pred=cat_model.predict(x_test)
print(accuracy_score(y_test,y_pred))
Importance=pd.DataFrame({'Importance':cat_model.feature_importances_},
            index=x_train.columns)
Importance
Importance.sort_values(by='Importance',
                      axis=0,
                      ascending=True).plot(kind='barh',color='b')
plt.show()
