# Import The Libraries

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import LabelBinarizer, StandardScaler

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.metrics import mean_squared_error , r2_score

from sklearn.model_selection import cross_val_score, GridSearchCV



from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
housing = pd.read_csv('../input/california-housing-prices/housing.csv')

housing.head()
# total_bedrooms attribute has nulls

housing.info()
housing.describe()
attributes = ['median_house_value', 'median_income',

             'total_rooms', 'housing_median_age'] 

             

plt.figure(figsize=(20, 20))             

sns.pairplot(housing[attributes])             

plt.show();
# good relationship between median_income and median_house_value

housing.plot(kind='scatter', x='median_income', y='median_house_value',

            alpha=0.1, figsize=(8,5))

plt.show()
# categorical attribute

housing['ocean_proximity'].value_counts()
housing['rooms_per_household'] = housing['total_rooms']/housing['households']

housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']

housing['population_per_household'] = housing['population']/housing['households']
housing.corr()['median_house_value'].sort_values(ascending=False)
housing.columns
attr_names = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',

               'total_bedrooms', 'population', 'households', 'median_income',

               'ocean_proximity', 'rooms_per_household',

               'bedrooms_per_room', 'population_per_household']



X = housing[attr_names]

y = housing['median_house_value']
# split Data 80% train , 20% test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
class DataFrameSelector(BaseEstimator, TransformerMixin):

    

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

        

    def fit(self, X, y=None):return self

    

    def transform(self, X):return X[self.attribute_names].values
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

# this component gives us the flexibility to add extra attributes to our pipeline



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    

    def __init__(self, add_bedrooms_per_room = True):

        self.add_bedrooms_per_room = add_bedrooms_per_room

    

    def fit(self, X, y=None):return self

    

    def transform(self, X, y=None):

        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]

        population_per_household = X[:, population_ix] / X[:, household_ix]

        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]



        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

        
class MyLabelBinarizer(TransformerMixin):

    

    def __init__(self, *args, **kwargs):

        self.encoder = LabelBinarizer(*args, **kwargs)

    

    def fit(self, x, y=0):

        self.encoder.fit(x)

        return self

    

    def transform(self, x, y=0):return self.encoder.transform(x)
num_attribs = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',

       'total_bedrooms', 'population', 'households', 'median_income']



cat_attribs = ["ocean_proximity"]
# our numerical pipeline

num_pipeline = Pipeline([

                    ('selector', DataFrameSelector(num_attribs)),

                    ('imputer', SimpleImputer(strategy="median")),

                    ('attribs_adder', CombinedAttributesAdder()),

                    ('std_scaler', StandardScaler()),

                ])
# our categorical pipeline

cat_pipeline = Pipeline([

    ('selector', DataFrameSelector(cat_attribs)),

    ('label_binarizer', MyLabelBinarizer()),

])
full_pipeline = FeatureUnion(transformer_list=[

    ('num_pipeline', num_pipeline),

    ('cat_pipeline', cat_pipeline),

])
X_train_prepared = full_pipeline.fit_transform(X_train)
# LinearRegression Model

lin_reg = LinearRegression()

scores = cross_val_score(lin_reg, X_train_prepared, y_train,

                        scoring="neg_mean_squared_error", cv=10)



rmse_scores = np.sqrt(-scores)

print("Mean:\t\t ", rmse_scores.mean(), "\nStandard Deviation:", rmse_scores.std())
# Decision Tree Regressor

tree_reg = DecisionTreeRegressor()

scores = cross_val_score(tree_reg, X_train_prepared, y_train,

                        scoring="neg_mean_squared_error", cv=10)



rmse_scores = np.sqrt(-scores)

print("Mean:\t\t ", rmse_scores.mean(), "\nStandard Deviation:", rmse_scores.std())
# Gradient Boosting Regressor

grad_reg = GradientBoostingRegressor()

scores = cross_val_score(grad_reg, X_train_prepared, y_train,

                               scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

print("Mean:\t\t ", rmse_scores.mean(), "\nStandard Deviation:", rmse_scores.std())
#Random Forest Regressor

forest_reg = RandomForestRegressor()

scores = cross_val_score(forest_reg, X_train_prepared, y_train,

                               scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

print("Mean:\t\t ", rmse_scores.mean(), "\nStandard Deviation:", rmse_scores.std())
param_grid = [

    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},

  ]
forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                          scoring='neg_mean_squared_error')

grid_search.fit(X_train_prepared, y_train)
cvres = grid_search.cv_results_

print("{}\t\t {}\n".format('Mean Score','Parameters'))

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    x = np.sqrt(-mean_score)

    y = params

    print("{:.2f}\t {}".format(x, y))

  
final_model = grid_search.best_estimator_



X_test_prepared = full_pipeline.transform(X_test)

y_pred = final_model.predict(X_test_prepared)



print('R-Squared:', r2_score(y_test, y_pred))

print("Root Mean Square Error:", np.sqrt(mean_squared_error(y_test, y_pred)))