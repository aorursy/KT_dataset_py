import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

df = pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')

df.head()
df.shape
print(df['Fuel_Type'].unique())

print(df['Seller_Type'].unique())

print(df['Transmission'].unique())

print(df['Owner'].unique())
### Check missing and null values

df.isnull().sum()
df.describe()
df.columns
final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven','Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]

final_dataset.head()
final_dataset['Current Year'] = 2020

final_dataset.head()
final_dataset['No. of Year'] = final_dataset['Current Year'] - final_dataset['Year']
final_dataset.head()
final_dataset.drop(columns = ['Year','Current Year'], axis=1,inplace=True)
final_dataset.head()
final_dataset = pd.get_dummies(final_dataset, drop_first=True)
final_dataset.head()
final_dataset.corr()
plt.figure(figsize=(15,15))

sns.pairplot(final_dataset)
plt.figure(figsize=(20,20))

sns.heatmap(final_dataset.corr(), annot=True, cmap='RdYlGn')
## Divide dataset into dependent and independent feature

X = final_dataset.iloc[:,1:]

y = final_dataset.iloc[:,0]

X.head()
y.head()
###Feature Importance

from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor()

model.fit(X,y)
print(model.feature_importances_)
# plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(5).plot(kind='barh')

plt.show()

# Divide X and Y feature into train and test data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2)
X_train.shape
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
## HyperParametres

n_estimators = [ int(x) for x in np.linspace(start=100, stop=1200, num=12)]

print(n_estimators)
## Randomized Search Cv



# Number of trees in random forest

n_estimators = [ int(x) for x in np.linspace(start=100, stop=1200, num=12)]

# Number of feature to consider every split

max_features = ['auto','sqrt']

# Maximum number of levels in Tree

max_depth = [int(x) for x in np.linspace(5,30, num=6)]

# Minimum number of samples required to split a node

min_samples_split = [2,5,10,15, 100]

# Minimum number of samples required at each leaf node

min_samples_leaf =[1,2,5,10]
from sklearn.model_selection import RandomizedSearchCV
# Create the randomgrid

random_grid = {

    'n_estimators': n_estimators,

    'max_features': max_features,

    'max_depth': max_depth,

    'min_samples_split': min_samples_split,

    'min_samples_leaf': min_samples_leaf,

}

print(random_grid)
# use to random grid search for best hypermeters

# First Create the base model to tune

rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator=rf,param_distributions=random_grid, scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=2,random_state=42, n_jobs=1)
rf_random.fit(X_train,y_train)
predictions = rf_random.predict(X_test)

predictions
sns.distplot(y_test - predictions)
plt.scatter(y_test, predictions)