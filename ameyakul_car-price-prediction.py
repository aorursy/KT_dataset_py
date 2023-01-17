import pandas as pd
df = pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')
df.head()
df.shape
# checking unique values of the categorical features

print(df['Seller_Type'].unique())

print(df['Transmission'].unique())

print(df['Fuel_Type'].unique())

print(df['Owner'].unique())
# checking null values or missing data

df.isnull().sum()
df.describe()
df['current year'] = 2020
df.head()
df.columns
final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',

       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner', 'current year']]
final_dataset.head()
final_dataset['no of year'] = final_dataset['current year'] - final_dataset['Year']
final_dataset.head()
final_dataset.drop(['Year'], axis=1, inplace=True)
final_dataset.drop(['current year'], axis=1, inplace=True)
final_dataset.head()
final_dataset=pd.get_dummies(final_dataset, drop_first=True)
final_dataset.head()
final_dataset.corr()
import seaborn as sns
sns.pairplot(final_dataset)
import matplotlib.pyplot as plt

%matplotlib inline
corrmat = final_dataset.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

# plot heatmap

g = sns.heatmap(final_dataset[top_corr_features].corr(), annot=True, cmap = 'RdYlGn')
final_dataset.head()
# independent feature

X = final_dataset.iloc[:, 1:]



# dependent feature

y = final_dataset.iloc[:, 0]
X.head()
y.head()
## Feature Importance

from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor()

model.fit(X,y)
print(model.feature_importances_)
# plotting the feature importances

feature_importance = pd.Series(model.feature_importances_, index=X.columns)

feature_importance.plot(kind='barh')

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train.shape

from sklearn.ensemble import RandomForestRegressor

rf_random = RandomForestRegressor()
import numpy as np
## Hyperparameters

n_estimators = [int(x) for x in np.linspace(start=100 ,stop=1200, num=12)]

print(n_estimators)
# Randomized search CV



# no of trees in random forest 

n_estimators = [int(x) for x in np.linspace(start=100 ,stop=1200, num=12)]



# No of features to consider at every split 

max_features = ['auto', 'sqrt']



# Maximum no of levels in a tree

max_depth = [int(x) for x in np.linspace(5, 30, num=6)]



# minimum number of samples required to split a node 

min_samples_split = [2, 5, 10, 15, 100]



# minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 5, 10]
from sklearn.model_selection import RandomizedSearchCV
# create a random grid 



random_grid = { 'n_estimators':n_estimators,

                'max_features': max_features,

                'max_depth': max_depth,

                'min_samples_split': min_samples_split,

                'min_samples_leaf': min_samples_leaf}

print(random_grid)
# Use the random grid to search for the best hyer parameter 

# First create the base model to tune

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)
rf_random.fit(X_train, y_train)
predictions = rf_random.predict(X_test)
predictions
sns.distplot(y_test-predictions)
plt.scatter(y_test, predictions)
import pickle



# open a file where you want to store the data

file = open('random_forest_regression_model.pkl', 'wb')



# dump info to that file

pickle.dump(rf_random, file)