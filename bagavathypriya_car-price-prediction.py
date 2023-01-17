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
df=pd.read_csv("../input/vehicle-dataset-from-cardekho/car data.csv")
df.head()
df.shape
print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())
df.isnull().sum()
df.describe()
df.columns
final_df=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
df['year']=2020
df.head()
df['no_of_yrs']=df['year']-df['Year']
df.head()
df.drop(['Year'],axis=1,inplace=True)
df.head()
df.drop(['year'],axis=1,inplace=True)
df.head()
df.drop(['Car_Name'],axis=1,inplace=True)
final_df=pd.get_dummies(df,drop_first=True)
final_df.head()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
sns.heatmap(final_df.corr())
final_df.head()
x=final_df.iloc[:,1:]
y=final_df.iloc[:,1]
y.head()
#Feature Importance
from sklearn.ensemble import ExtraTreesRegressor
mod=ExtraTreesRegressor()
mod.fit(x,y)
print(mod.feature_importances_)
#plot graph of feature importances for better visualization 
feat_importances = pd.Series(mod.feature_importances_, index=x.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
print(xtrain.shape)
print(xtest.shape)
from sklearn.ensemble import RandomForestRegressor
rfreg=RandomForestRegressor()
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)
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
rf_random = RandomizedSearchCV(estimator = rfreg, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(xtrain,ytrain)
rf_random.best_params_
rf_random.best_score_
predictions=rf_random.predict(xtest)
sns.distplot(ytest-predictions)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(ytest, predictions))
print('MSE:', metrics.mean_squared_error(ytest, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(ytest, predictions)))
import pickle
# open a file, where you ant to store the data
file = open('model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)
