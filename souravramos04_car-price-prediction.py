

import pandas as pd
df = pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')
df.head()
df.shape
## chekc missing or null values

df.isnull().sum()
df.isnull().any()
print(df['Fuel_Type'].unique())

print(df['Seller_Type'].unique())

print(df['Transmission'].unique())

print(df['Owner'].unique())
df.describe()
df.columns
final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',

       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_dataset.shape
final_dataset.head()
final_dataset['Current_Year'] = 2020
final_dataset.head()
final_dataset['no_year'] = final_dataset['Current_Year']-final_dataset['Year']
#for i in range(len(df['Year'])):

#    df['Year'][i] = 2020 - df['Year'][i]
final_dataset.head()
final_dataset.drop(['Year', 'Current_Year'], axis=1, inplace=True)
final_dataset.head()
final_dataset = pd.get_dummies(final_dataset, drop_first=True)
final_dataset.shape
final_dataset.head()
final_dataset.corr()
!pip install seaborn

import seaborn as sns
sns.pairplot(final_dataset)
import matplotlib.pyplot as plt

%matplotlib inline
corrmat = final_dataset.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

# plot heat map

g = sns.heatmap(final_dataset[top_corr_features].corr(), annot=True, cmap='RdYlGn')
top_corr_features
# dependent and independent features.

X = final_dataset.iloc[:, 1:]

y = final_dataset.iloc[:, 0]
X.head()
y.head()
# feature importance

from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor()

model.fit(X,y)
print(model.feature_importances_)
# plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(5).plot(kind='barh')

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
from sklearn.ensemble import RandomForestRegressor

rf_random = RandomForestRegressor()
import numpy as np
### hyperparameters 



# number of trees in random forrest

n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]



# number of features to consider at every split

max_features = ['auto', 'sqrt']



# maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(5, 30, num=6)]



# minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 100]



# minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 5, 10]
max_depth
from sklearn.model_selection import RandomizedSearchCV
# create the random grid

random_grid = {

                'n_estimators' : n_estimators,

                'max_features' : max_features,

                'max_depth' : max_depth,

                'min_samples_split' : min_samples_split,

                'min_samples_leaf' : min_samples_leaf

}

print(random_grid)
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(

        estimator = rf, 

        param_distributions = random_grid,

        scoring = 'neg_mean_squared_error',

        n_iter = 10,

        cv = 5,

        verbose = 2,

        random_state = 42,

        n_jobs = 1

)
rf_random.fit(X_train, y_train)
predictions = rf_random.predict(X_test)
predictions
sns.distplot(y_test-predictions)
plt.scatter(y_test, predictions)
import pickle

# open a file, where you want to store the data

file = open('random_forest_regression_model1.pkl', 'wb')



# dump information to that file

pickle.dump(rf_random, file)