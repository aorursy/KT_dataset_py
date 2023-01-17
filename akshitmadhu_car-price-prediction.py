# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error,mean_squared_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Loading the dataset
df = pd.read_csv('../input/vehicle-dataset-from-cardekho/CAR DETAILS FROM CAR DEKHO.csv')
df.head()
df.shape
df.dtypes
df.describe()
df.isnull().sum()
df.columns
final_df = df[['year', 'selling_price', 'km_driven', 'fuel', 'seller_type',
       'transmission', 'owner']]
final_df['owner'].value_counts()
final_df.drop(final_df[final_df['owner']=='Test Drive Car'].index,axis=0,inplace=True)
final_df['No_of_previous_owner'] = final_df['owner'].map({'First Owner':1,'Second Owner':2,'Third Owner':3,"Fourth & Above":4})
final_df.drop('owner',axis=1,inplace=True)
final_df = final_df.dropna()
final_df['No_of_previous_owner'] = final_df['No_of_previous_owner'].astype(int)
final_df.head()
final_df['seller_type'].value_counts()
final_df.drop(final_df[final_df['seller_type']=='Trustmark Dealer'].index,axis=0,inplace=True)
final_df['Current Year'] = 2020
final_df['No_of_Years'] = final_df['Current Year'] - final_df['year']
final_df.drop(['year','Current Year'],axis=1,inplace=True)
final_df = pd.get_dummies(final_df,drop_first=True)
final_df.shape
final_df.head()
final_df.dtypes
# Seprating the dependent variable and target variable
X = final_df.drop('selling_price',axis=1)
y = final_df['selling_price']
# Splitting training and testing data
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=5)
# Transforming the data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# Building a machine learning model using Random Forest Regressor
regressor = RandomForestRegressor()
# Hyperparameter optimization using Randomized Search CV
from sklearn.model_selection import RandomizedSearchCV


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
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
final_df.isnull().sum()
# Training the data
rf_random.fit(x_train,y_train)
print("Best Parameters:-",rf_random.best_params_)
print("Best Score: ",rf_random.best_score_)
import math
y_pred = rf_random.predict(x_test)
y_pred = y_pred.astype(int)
sns.distplot(y_test-y_pred)
plt.show()
print('MAE:',mean_absolute_error(y_test,y_pred))
print('MSE:', mean_squared_error(y_test,y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test,y_pred)))
