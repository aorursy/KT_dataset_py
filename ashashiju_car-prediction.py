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
df=pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')

df.head()
df.isnull().sum()
finalDS=df[['Year','Selling_Price','Present_Price' ,'Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
finalDS['CurrentYear']=2020
finalDS['noOfYears']=finalDS['CurrentYear']-finalDS['Year']
finalDS.drop(['Year','CurrentYear'],axis=1,inplace=True)

finalDS.head()
finalDS=  pd.get_dummies(finalDS,drop_first=True)
finalDS.corr()
import seaborn as sns

import matplotlib.pyplot  as plt

%matplotlib inline

corrmat=finalDS.corr()

topCorrFeatures=corrmat.index

plt.figure(figsize=(20,20))

g=sns.heatmap(finalDS[topCorrFeatures].corr(),annot=True,cmap="RdYlGn")
x=finalDS.iloc[:,1:]

y=finalDS.iloc[:,0]
from sklearn.ensemble import ExtraTreesRegressor

model=ExtraTreesRegressor()

model.fit(x,y)

print(model.feature_importances_)
feature_importances=pd.Series(model.feature_importances_,index=x.columns)

feature_importances.nlargest(5).plot(kind='barh')

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestRegressor

regressor=RandomForestRegressor()
#Randomized Search CV

from sklearn.model_selection import RandomizedSearchCV

import numpy as np



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

# max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 5, 10]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}



print(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestRegressor()
# search across 100 different combinations

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
rf_random.best_score_
predictions=rf_random.predict(X_test)
sns.distplot(y_test-predictions)
from sklearn import metrics





print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
import pickle

# open a file, where you ant to store the data

file = open('random_forest_regression_model.pkl', 'wb')



# dump information to that file

pickle.dump(rf_random, file)