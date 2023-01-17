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

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit

from pandas.plotting import scatter_matrix

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
df = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')

df.head()
df.info()
df["ocean_proximity"].value_counts()
df.describe()
%matplotlib inline

df.hist(bins=50, figsize=(20,15))

plt.show()
train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
print(len(train_set))

print(len(test_set))
df["income_cat"] = pd.cut(df["median_income"],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])
df["income_cat"].hist()
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2 ,random_state = 42)
for train_index, test_index in split.split(df,df["income_cat"]):

    strat_train_set = df.loc[train_index]

    strat_test_set = df.loc[test_index]
strat_test_set["income_cat"].value_counts()/len(strat_test_set)
df.head()
for set_ in (strat_train_set,strat_test_set):

    set_.drop("income_cat",axis=1,inplace=True)
df.head()
df = strat_train_set.copy()
df.head()
df.plot(kind="scatter", x="longitude", y="latitude")
df.plot(kind="scatter" ,x="longitude", y="latitude" ,alpha=0.1)
df.plot(kind="scatter" ,x="longitude" ,y="latitude" ,alpha = 0.4,

       s=df["population"]/100, label="population", figsize=(10,7),

       c="median_house_value" ,cmap=plt.get_cmap("jet"), colorbar=True)

plt.legend()
corr_Matrix = df.corr()
corr_Matrix["median_house_value"].sort_values(ascending = False)
attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(df[attributes], figsize = (12,8))
df.plot(kind="scatter" , x = "median_income", y="median_house_value" ,alpha = 0.1)
df["rooms_per_household"] = df["total_rooms"]/df["households"]

df["bedrooms_per_room"] = df["total_bedrooms"]/df["total_rooms"]

df["poplation_per_household"] = df["population"]/df["households"]
corr_Matrix = df.corr()

corr_Matrix["median_house_value"].sort_values(ascending = False)
df = strat_train_set.drop("median_house_value",axis=1)

df_labels = strat_train_set["median_house_value"].copy()
df.head()
df_labels.head()
imputer = SimpleImputer(strategy="median")

df_num = df.drop("ocean_proximity",axis=1)

imputer.fit(df_num)

imputer.statistics_

df_num.median().values
x = imputer.transform(df_num)
housing_tr = pd.DataFrame(x,columns=df_num.columns)
df_cat = df[["ocean_proximity"]]
df_cat.head(10)
ordinal_encoder = OrdinalEncoder()

df_cat_encoded = ordinal_encoder.fit_transform(df_cat)

df_cat_encoded[:10]
ordinal_encoder.categories_
cat_encoder = OneHotEncoder()

df_cat_1hot = cat_encoder.fit_transform(df_cat)

df_cat_1hot
df_cat_1hot.toarray()
cat_encoder.categories_
from sklearn.base import BaseEstimator, TransformerMixin



# column index

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs

        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):

        return self  # nothing else to do

    def transform(self, X):

        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]

        population_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

            return np.c_[X, rooms_per_household, population_per_household,

                         bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household, population_per_household]



attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

df_extra_attribs = attr_adder.transform(df.values)
df_extra_attribs = pd.DataFrame(

    df_extra_attribs,

    columns=list(df.columns)+["rooms_per_household", "population_per_household"],

    index=df.index)

df_extra_attribs.head()
num_pipeline =  Pipeline([

    ('imputer',SimpleImputer(strategy = "median")),

    ("attribs_adder",CombinedAttributesAdder()),

    ('std_scaler',StandardScaler()),

])

df_num_tr = num_pipeline.fit_transform(df_num)
num_attribs = list(df_num)

cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([

    ("num",num_pipeline,num_attribs),

    ("cat",OneHotEncoder(),cat_attribs),

])

df_prepared = full_pipeline.fit_transform(df)

df_prepared[:5]
lin_reg = LinearRegression()

lin_reg.fit(df_prepared,df_labels)
some_data = df.iloc[:5]

some_labels = df_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)
print("Prediction:",lin_reg.predict(some_data_prepared))
print("Labels:",list(some_labels))
some_data_prepared
df_predictions = lin_reg.predict(df_prepared)
lin_mse = mean_squared_error(df_labels,df_predictions)

lin_rmse = np.sqrt(lin_mse)
lin_rmse
tree_reg = DecisionTreeRegressor()

tree_reg.fit(df_prepared,df_labels)
df_treepredictions = tree_reg.predict(df_prepared)
tree_mse = mean_squared_error(df_labels,df_treepredictions)

tree_rmse = np.sqrt(tree_mse)

tree_rmse
#Cross Validation

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg,df_prepared,df_labels,scoring = "neg_mean_squared_error",cv=10)

tree_rmse_score = np.sqrt(-scores)
tree_rmse_score
def display_scores(scores):

    print("Scores:",scores)

    print("Mean:",scores.mean())

    print("standard devition",scores.std())
display_scores(tree_rmse_score)
lin_scores = cross_val_score(lin_reg,df_prepared,df_labels,scoring="neg_mean_squared_error",cv=10)

lin_rmse_score = np.sqrt(-lin_scores)

display_scores(lin_rmse_score)
#using random forest regressor

forest_reg = RandomForestRegressor()
forest_reg.fit(df_prepared,df_labels)
df_forest_predictions = forest_reg.predict(df_prepared)
#calculating rmse

forest_mse = mean_squared_error(df_labels,df_forest_predictions)

forest_rmse = np.sqrt(forest_mse)

forest_rmse
#apllying cross validation

forest_scores = cross_val_score(forest_reg,df_prepared,df_labels,scoring="neg_mean_squared_error",cv=10)
tree_rmse_score = np.sqrt(-forest_scores)
display_scores(tree_rmse_score)
#Fine tuning

param_grid = [

    {'n_estimators':[3,10,30],'max_features':[2,4,6,8]},

    {'bootstrap':[False],'n_estimators': [3,10],'max_features':[2,3,4]},

]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg,param_grid,cv=5,scoring="neg_mean_squared_error",

                           return_train_score=True



)

grid_search.fit(df_prepared,df_labels)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
#feature importance



feature_importance = grid_search.best_estimator_.feature_importances_

feature_importance
extra_attribs = ["rooms_per_hhold","pop_per_hhold","bedrooms_per_room"]

cat_encoder = full_pipeline.named_transformers_["cat"]

cat_one_hot_attribs = list(cat_encoder.categories_[0])

attribs = num_attribs + extra_attribs + cat_one_hot_attribs

sorted(zip(feature_importance,attribs),reverse=True)
#evaluate on test set 

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value" ,axis=1 )

y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)
final_rmse
from scipy import stats

confidence = 0.95

squared_errors = (final_predictions - y_test) ** 2

np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,loc=squared_errors.mean(),scale=stats.sem(squared_errors)))