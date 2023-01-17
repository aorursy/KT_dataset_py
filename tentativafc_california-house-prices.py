# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import matplotlib.pyplot as plt



plt.style.use('seaborn')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

import warnings

warnings.filterwarnings('ignore')        

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')

df
df.describe()
df.hist(bins=50, figsize=(20,15))
df['income_cat'] = np.ceil(df["median_income"] / 1.5)

#df['income_cat'].where(df["income_cat"] < 5, 5.0, inplace=True)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



for train_index, test_index in split.split(df, df['income_cat']):

    strat_train_set = df.loc[train_index]

    strat_test_set = df.loc[test_index]

    

for set_ in (strat_train_set, strat_test_set):

    set_.drop('income_cat', axis=1, inplace=True)

    
df_complete = strat_train_set.copy()



df = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set

housing_labels = strat_train_set["median_house_value"].copy()
df_complete.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,s=df_complete["population"]/100, label="population", figsize=(10,7),c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,sharex=False)
#from pandas.plotting import scatter_matrix



#attributes = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','median_house_value']

#scatter_matrix(df[attributes])



import seaborn as sns

sns.set(style="ticks")

sns.pairplot(df)
from sklearn.base import BaseEstimator, TransformerMixin



# get the right column indices: safer than hard-coding indices 3, 4, 5, 6

rooms_ix, bedrooms_ix, population_ix, household_ix = [list(df.columns).index(col) for col in ("total_rooms", "total_bedrooms", "population", "households")]


from sklearn.preprocessing import FunctionTransformer



def add_extra_features(X, add_bedrooms_per_room=True):

    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]

    population_per_household = X[:, population_ix] / X[:, household_ix]

    if add_bedrooms_per_room:

        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

        return np.c_[X, rooms_per_household, population_per_household,

                     bedrooms_per_room]

    else:

        return np.c_[X, rooms_per_household, population_per_household]
def apply_log1p(X):

     attr = [household_ix,population_ix,bedrooms_ix,rooms_ix,8,9,10]

     for idx in attr:

        X[:,idx] = np.log1p(X[:,idx])

     return X

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),

        ('apply_log', FunctionTransformer(apply_log1p, validate=False)),

        ('std_scaler', StandardScaler()),

    ])



df_num = df.drop('ocean_proximity', axis=1)

df_num_tr = num_pipeline.fit_transform(df_num)
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder



num_attribs = list(df_num)

cat_attribs = ["ocean_proximity"]



full_pipeline = ColumnTransformer([

    ("num", num_pipeline, num_attribs),

    ("cat", OneHotEncoder(), cat_attribs),

])



housing_prepared = full_pipeline.fit_transform(df)

housing_prepared_df = pd.DataFrame(housing_prepared)





cols = [col for col in df.columns[0:8]]



# new columns

cols.extend(['rooms_per_household','population_per_household','bedrooms_per_room', 'nearby_lt_1H_OCEAN','nearby_INLAND','nearby_ISLAND', 'nearby_NEARBAY', 'nearby0 n1NEAR_OCEAN'])

housing_prepared_df.columns = cols



housing_prepared_df.hist(bins=50, figsize=(20,15))
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor(random_state=42)

tree_reg.fit(housing_prepared, housing_labels)
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor()

forest_reg.fit(housing_prepared, housing_labels)
from sklearn.model_selection import cross_val_score



def display_scores(scores):

    print('Scores:',scores)

    print('Mean:',scores.mean())

    print('Standard deviation:',scores.std())

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,scoring='r2', cv=10)

tree_r2_scores = scores

display_scores(tree_r2_scores)
scores = cross_val_score(lin_reg, housing_prepared, housing_labels,scoring='r2', cv=10)

lin_r2_scores = scores

display_scores(lin_r2_scores)
scores = cross_val_score(forest_reg, housing_prepared, housing_labels,scoring='r2', cv=10)

forest_r2_scores = scores

display_scores(forest_r2_scores)
X_test = strat_test_set.drop("median_house_value", axis=1)

y_test = strat_test_set["median_house_value"].copy()



X_test_prepared = full_pipeline.transform(X_test)



testing_params_scores = []

for x in range(2,20):

    dt_reg = DecisionTreeRegressor(max_depth=x)

    reg_tree = dt_reg.fit(housing_prepared, housing_labels)

    score = reg_tree.score(X_test_prepared, y_test)

    testing_params_scores.append((x, score))





df_scores = pd.DataFrame(testing_params_scores)

df_scores.columns = ['depth', 'score']

df_scores.sort_values(by='score', ascending=False)
df_scores.plot.line(x='depth', y='score', c='DarkBlue')
from sklearn.metrics import r2_score



X_test = strat_test_set.drop("median_house_value", axis=1)

y_test = strat_test_set["median_house_value"].copy()



X_test_prepared = full_pipeline.transform(X_test)

final_predictions = lin_reg.predict(X_test_prepared)



lin_final_r2 = r2_score(y_test, final_predictions)

lin_final_r2
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint



param_distribs = {

        'max_depth': randint(low=2, high=50),

    }



rnd_search = RandomizedSearchCV(tree_reg, param_distributions=param_distribs,

                                n_iter=10, cv=5, scoring='r2', random_state=42)

rnd_search.fit(housing_prepared, housing_labels)

best_tree = rnd_search.best_estimator_
final_model = rnd_search.best_estimator_



X_test = strat_test_set.drop("median_house_value", axis=1)

y_test = strat_test_set["median_house_value"].copy()



X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)



tree_final_r2 = r2_score(y_test, final_predictions)

tree_final_r2
param_distribs = {

        'n_estimators': randint(low=1, high=200),

        'max_features': randint(low=1, high=8),

    }



rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,

                                n_iter=10, cv=5, scoring='r2', random_state=42)

rnd_search.fit(housing_prepared, housing_labels)



rnd_search.best_estimator_
final_model = rnd_search.best_estimator_



X_test = strat_test_set.drop("median_house_value", axis=1)

y_test = strat_test_set["median_house_value"].copy()



X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)



forest_final_r2 = r2_score(y_test, final_predictions)

forest_final_r2
d = {'model':['linear_regression','regression_tree','random_forest'], 'r2_test':[lin_final_r2,tree_final_r2,forest_final_r2]}

df_res = pd.DataFrame(data=d)

df_res.sort_values(by='r2_test',ascending=False)
from sklearn.tree import export_graphviz



feature_names = cols

label_name = ['median_house_value']

    

export_graphviz(best_tree, out_file='tree.dot', rounded = True, proportion = True, precision = 2, filled = True, feature_names = feature_names, class_names = label_name)



from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



# # Display in jupyter notebook

from IPython.display import Image

Image(filename = 'tree.png')
from sklearn.tree import _tree



def tree_to_code(tree, feature_names):

    tree_ = tree.tree_

    feature_name = [

        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"

        for i in tree_.feature

    ]

    

    def recurse(node, depth):

        indent = "  " * depth

        if tree_.feature[node] != _tree.TREE_UNDEFINED:

            name = feature_name[node]

            threshold = tree_.threshold[node]

            print("{}se {} <= {}:".format(indent, name, threshold))

            recurse(tree_.children_left[node], depth + 1)

            print("{}senao:  # se {} > {}".format(indent, name, threshold))

            recurse(tree_.children_right[node], depth + 1)

        else:

            print("{} valor {}".format(indent, tree_.value[node]))



    recurse(0, 1)

    

tree_to_code(best_tree, feature_names)