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
sns.set_style('whitegrid')
df = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')
df.head()
df.drop(['Serial No.'] , axis=1, inplace=True)
df.head()
df.describe()
df.shape
sns.pairplot(df)
df_corr = df.corr()
sns.heatmap(df_corr ,cmap='viridis_r' ,annot=True)
y = df['Chance of Admit ']
X = df.drop(['Chance of Admit '],axis =1)
X.shape
y.shape
from sklearn.model_selection import train_test_split
X_train ,X_test , y_train ,y_test = train_test_split(X,y,test_size=0.2,random_state=101)
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso,Ridge,ElasticNet,LinearRegression,LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

models = [['DecisionTree :',DecisionTreeRegressor()],
           ['Linear Regression :', LinearRegression()],
           ['RandomForest :',RandomForestRegressor()],
           ['KNeighbours :', KNeighborsRegressor(n_neighbors = 2)],
           ['SVM :', SVR()],
           ['AdaBoostClassifier :', AdaBoostRegressor()],
           ['GradientBoostingClassifier: ', GradientBoostingRegressor()],
           ['Xgboost: ', XGBRegressor()],
           ['Lasso: ', Lasso()],
           ['Ridge: ', Ridge()],
           ['ElasticNet: ', ElasticNet()]]

print("""
Loading Results...
Grab A Snack
    """ )


for name,model in models:
    model = model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('RMSE Score for :',name, (np.sqrt(mean_squared_error(y_test, predictions))))
    print('R- Squared Score for: ',name, (r2_score(y_test, predictions)))
    print('\n')
rfr = RandomForestRegressor()
rfr.fit(X,y)

ImportantVar = pd.DataFrame()
ImportantVar['Features'] = X.columns
ImportantVar['Importance'] = rfr.feature_importances_
ImportantVar.sort_values('Importance' ,ascending=False)