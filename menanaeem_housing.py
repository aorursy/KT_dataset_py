import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
housing = pd.read_csv('../input/housing.csv')
housing.head()
housing.shape
housing.describe(include='all')
housing['ocean_proximity'].value_counts()
housing.info()
housing.hist(figsize=(20,20))
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
train_set.shape
test_set.shape
df = train_set.copy()
df.plot(kind='scatter',x='longitude',y='latitude',alpha=0.2)
df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

    s=housing["population"]/100, label="population", figsize=(10,7),

    c="median_house_value", cmap=plt.get_cmap("Reds"), colorbar=True,

    sharex=False)
corr_matrix = df.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
df.head()
df['room_per_house'] = df['total_rooms']/df['households']

df['bedrooms_per_house'] = df['total_bedrooms']/df['households']

df['population_per_house'] = df['population']/df['households']
corr_matrix = df.corr()

corr_matrix['median_house_value'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12, 8))

df.describe()
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler



from sklearn.base import BaseEstimator, TransformerMixin



from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer









housing = train_set.drop("median_house_value", axis=1) # drop labels for training set

housing_labels = train_set["median_house_value"].copy() # the labels

housing_num = housing.drop('ocean_proximity', axis=1) # dropping the categoral values so we have only numbers

housing_cat = housing[['ocean_proximity']] # the categoral values


rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6



# calss to add two attributes [ rooms_per_household , population_per_household ]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin): 

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs

        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):

        return self  # nothing else to do

    def transform(self, X, y=None):

        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]

        population_per_household = X[:, population_ix] / X[:, household_ix]

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

            return np.c_[X, rooms_per_household, population_per_household,

                         bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household, population_per_household]



attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

housing_extra_attribs = attr_adder.transform(housing.values)
num_pipeline = Pipeline([ # this pipeline will work on the numerical part only 

        ('imputer', SimpleImputer(strategy="median")), # filling null values with the median 

        ('attribs_adder', CombinedAttributesAdder()), # adding 2 extra attributes discussed earlier

        ('std_scaler', StandardScaler()), # scalling the numberical values to work better in the machine larning algorithm

    ])





num_attribs = list(housing_num)

cat_attribs = ["ocean_proximity"]



full_pipeline = ColumnTransformer([#here we apply the previous pipe line on numerical attruibutes and onehot encoded on categoral

        ("num", num_pipeline, num_attribs),

        ("cat", OneHotEncoder(), cat_attribs),

    ])



housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared.shape
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR
reg = LinearRegression()

ran = RandomForestRegressor()

svm_reg = SVR(kernel = "linear")
reg.fit(housing_prepared,housing_labels)

ran.fit(housing_prepared,housing_labels)

svm_reg.fit(housing_prepared,housing_labels)
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score
np.sqrt(mean_squared_error(reg.predict(housing_prepared),housing_labels))
np.sqrt(mean_squared_error(ran.predict(housing_prepared),housing_labels))
np.sqrt(mean_squared_error(svm_reg.predict(housing_prepared),housing_labels))
reg1 = cross_val_score(reg,housing_prepared,housing_labels,cv=10,scoring="neg_mean_squared_error")

ran1 = cross_val_score(ran,housing_prepared,housing_labels,cv=10,scoring="neg_mean_squared_error")

svm_reg1 = cross_val_score(svm_reg,housing_prepared,housing_labels,cv=10,scoring="neg_mean_squared_error")
np.mean(np.sqrt(-reg1))
np.mean(np.sqrt(-ran1))
np.mean(np.sqrt(-svm_reg1))
from sklearn.model_selection import GridSearchCV



param_grid = [

    # try 12 (3×4) combinations of hyperparameters

    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

    # then try 6 (2×3) combinations with bootstrap set as False

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},

  ]



forest_reg = RandomForestRegressor(random_state=42)

# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                           scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)
np.sqrt(mean_squared_error(grid_search.predict(housing_prepared),housing_labels))
final_model = grid_search.best_estimator_



X_test = test_set.drop("median_house_value", axis=1)

y_test = test_set["median_house_value"].copy()



X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)



final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)
final_rmse