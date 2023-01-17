# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.datasets import load_boston
df = load_boston()
# Let's check the df



df
# Let's make dataframe out of this data



data = pd.DataFrame(df.data,columns = df.feature_names)
# Let's check the head of the dataframe



data.head()
data['PRICE'] = df.target
# Let's separate feature variables and target variable



X = data.drop('PRICE',axis = 1)

y = data['PRICE']
# Let's check out the shape of our dataframe



data.shape
# Let's spit the data into train and test set



from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,

                                                 test_size = 0.25,

                                                random_state = 129)



X_train.shape
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression



lin_regressor=LinearRegression()

mse=cross_val_score(lin_regressor,X_train,y_train,scoring='neg_mean_squared_error',cv=5)

mean_mse=np.mean(mse)

print(mean_mse)
from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV



ridge=Ridge()

parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}

ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)

ridge_regressor.fit(X_train,y_train)

# Let's check out the best parameter and the score



print(ridge_regressor.best_params_)

print(ridge_regressor.best_score_)
from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

lasso=Lasso()

parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}

lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)



lasso_regressor.fit(X_train,y_train)

# Let's check out the best parameter and best score



print(lasso_regressor.best_params_)

print(lasso_regressor.best_score_)
#Let's make prediction



prediction_lasso=lasso_regressor.predict(X_test)

prediction_ridge=ridge_regressor.predict(X_test)
import seaborn as sns



sns.distplot(y_test-prediction_lasso)
import seaborn as sns



sns.distplot(y_test-prediction_ridge)