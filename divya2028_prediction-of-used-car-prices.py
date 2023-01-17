import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/vehicle-dataset-from-cardekho/car data.csv")
data.head()
data.shape
print(data['Seller_Type'].unique())
print(data['Transmission'].unique())
print(data['Owner'].unique())
print(data['Fuel_Type'].unique())
## check missing or null values
data.isnull().sum()
data.describe()
data.columns
final_data = data[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_data.head()
final_data['Current_Year'] = 2020
final_data.head()
final_data['No_of_Years'] = final_data['Current_Year']-final_data['Year']
final_data.head()
final_data.drop(['Year'], axis = 1, inplace = True)
final_data.drop(['Current_Year'], axis=1, inplace=True)
final_data.head()
final_data = pd.get_dummies(final_data, drop_first=True)
final_data.head()
final_data.corr()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.pairplot(final_data)
corrmat = final_data.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(final_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
X = final_data.iloc[:,1:] #independent features
y = final_data.iloc[:, 0] #dependent features
X.head()
y.head()
# feature importance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)
X_train.shape
from sklearn.ensemble import RandomForestRegressor
rf_random = RandomForestRegressor()
#Randomized Search CV

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
from sklearn.model_selection import RandomizedSearchCV
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)
predictions=rf_random.predict(X_test)
sns.distplot(y_test-predictions)
plt.scatter(y_test,predictions)
import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)