# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data
data=data[data['status']=='Placed']
data
data.info()
data.describe(include='all')
new_data=data[data['salary']<=600000]
X=data.drop(columns=['sl_no','status','salary'])
y=data['salary']
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)
ohe=OneHotEncoder()
ohe.fit(X[['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation']])
ohe.categories_
col_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation']),
                                 remainder='passthrough')
lr=LinearRegression()
pipe=make_pipeline(col_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2_score(y_pred,y_test)
y_test
y_pred
sns.boxplot(x='gender',y='salary',data=new_data)
sns.relplot(x='ssc_p',y='salary',data=new_data)
sns.boxplot(x='ssc_b',y='salary',data=new_data)
sns.relplot(x='ssc_p',y='salary',hue='hsc_b',style='hsc_s',data=new_data)
sns.relplot(x='degree_p',y='salary',hue='degree_t',style='workex',data=new_data)
sns.relplot(x='etest_p',y='salary',hue='specialisation',data=new_data)
sns.relplot(x='mba_p',y='salary',hue='specialisation',data=new_data)
X=new_data.drop(columns=['sl_no','status','salary','ssc_b'])
y=new_data['salary']
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder,PolynomialFeatures
from sklearn.metrics import r2_score
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)
ohe=OneHotEncoder()
ohe.fit(X[['gender','hsc_b','hsc_s','degree_t','workex','specialisation']])
ohe.categories_
col_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['gender','hsc_b','hsc_s','degree_t','workex','specialisation']),
                                  #(PolynomialFeatures(2),['ssc_p','hsc_p','degree_p','etest_p','mba_p']),
                                 remainder='passthrough')
lr=LinearRegression()
pipe=make_pipeline(col_trans,lr)
pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)
r2_score(y_pred,y_test)
y_test
y_pred
