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
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('../input/vehicle-dataset-from-cardekho/CAR DETAILS FROM CAR DEKHO.csv')

df.head()
df.shape
#Finding Unique values from dataset for categorical data
print(df['seller_type'].unique())
print(df['transmission'].unique())
print(df['owner'].unique())
print(df['fuel'].unique())
#checking missing or null values

df.isnull().sum()
df.describe()
df.columns
#Making final dataset
final_dataset = df[[ 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type',
       'transmission', 'owner']]

final_dataset.head()

#this is for adding another feature 'No of years'
final_dataset['Current_Year'] = 2020
final_dataset.head()
#creating another derived feature
final_dataset['No_of_years'] = final_dataset['Current_Year'] - final_dataset['year']
final_dataset.head()
#removing unwanted features i.e year and current_year

final_dataset.drop(['year'],axis=1,inplace=True)

final_dataset.drop(['Current_Year'],axis=1,inplace=True)
final_dataset.head()
final_dataset.head()
#One hot encoding --dummy variable trap..
final_dataset = pd.get_dummies(final_dataset,drop_first=True)
final_dataset.head()
#visualzation 
sns.pairplot(final_dataset)
%matplotlib inline

corrmat = final_dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))

#ploting heat map

g = sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")

final_dataset.head()
X = final_dataset.iloc[:,1:]
y = final_dataset.iloc[:,0]
X.head()
y.head()
#feature importance 

from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)
Feat_Importance = pd.Series(model.feature_importances_,index=X.columns)
Feat_Importance.nlargest(5).plot(kind='barh')
plt.show()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)
X_train.shape
#implementing randomforest regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
#Hyperparameters tuning using randomised search cv


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100 , stop = 1200 , num = 12)]
# Number of features to consider at every split
max_features = ['auto' , 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5 , 30 , num =6)]

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
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
from sklearn.model_selection import RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)

rf_random.best_params_
rf_random.best_score_
predictions = rf_random.predict(X_test)
sns.distplot(predictions - y_test)
plt.scatter(y_test,predictions)

import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

pickle.dump(rf_random,file)




