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
df=pd.read_csv('/kaggle/input/insurance/insurance.csv')
df.head()
df.info()
df['region'].unique()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#visualising the dependency of region on insurance
sns.barplot(x='region',y='charges',data=df)
sns.boxplot(x='charges',data=df)
sns.boxplot(x='charges',y='region',data=df)
df.hist()
#correlation between the columns
plt.rcParams['figure.figsize']=(12,8)
corr=df.corr()
sns.heatmap(corr,fmt='0.2f',annot=True,cmap=plt.cm.Blues)
#converting the object dtypes to int64
df['sex']=df['sex'].map({'male':0,'female':1})
df['smoker']=df['smoker'].map({'yes':1,'no':0})
df['region']=df['region'].map({'southwest':1,'southeast':2,'northwest':3,'northeast':4})
#columns and labels from the dataset
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
lr=LinearRegression()
lr.fit(X_train,y_train)

preds=lr.predict(X_test)
print('mean absolute error:',mean_absolute_error(preds,y_test))
results=pd.DataFrame({'y_test':y_test.values,'predictions':preds})
results.head()
lr.score(X_train,y_train)
#Scaling the data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit(X_test)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(X_train,y_train)
dt.score(X_train,y_train)
dt_preds=dt.predict(X_test)
print('mean squared error',mean_squared_error(dt_preds,y_test))
print('mean absolute error',mean_absolute_error(dt_preds,y_test))
from sklearn.model_selection import GridSearchCV
params={'max_depth':np.arange(2,10),'min_samples_leaf':np.arange(2,8)}
dt_best=GridSearchCV(estimator=dt,param_grid=params,verbose=1,cv=5)
dt_best.fit(X_train,y_train)
dt_best.best_params_,dt_best.best_score_
preds_best=dt_best.predict(X_test)
print('mean squared error',mean_squared_error(preds_best,y_test))
print('mean absolute error',mean_absolute_error(preds_best,y_test))

















