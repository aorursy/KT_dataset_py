# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import librareis:
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# importb dataset:
df=pd.read_csv("../input/Advertising.csv")
df.head()
df=df.drop('Unnamed: 0',axis=1)
df.head()
# visualization:
sns.scatterplot(df['TV'],df['Sales'],data=df,)
sns.scatterplot(df['Newspaper'],df['Sales'],data=df)
sns.scatterplot(df['Radio'],df['Sales'],data=df)
sns.pairplot(df)
sns.lmplot('TV','Sales',data=df)
sns.lmplot('Radio','Sales',data=df)
sns.lmplot('Newspaper','Sales',data=df)
# spliting the dataset:
X=df.iloc[:,:-1].values
y=df.iloc[:,3].values.reshape(-1,1)
X
y
# modeeling the dataset:
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
MSEs=cross_val_score(regressor,X,y,scoring='neg_mean_squared_error',cv=5)
mean_MSE=np.mean(MSEs)
print(mean_MSE)

# spliting the dataset  set on train and test set:
#from sklearn.model_selection import  train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# now the Ridge regression:
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X,y)
print(ridge_regressor.best_score_)
print(ridge_regressor.best_params_)
# now we imlement of Lasso:
from sklearn.linear_model import Lasso
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring ='neg_mean_squared_error',cv=5)
lasso_regressor.fit(X,y)

print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)
