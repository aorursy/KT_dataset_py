# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are aavailable in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import requests
import os
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt


RANDOM_STATE = 75
# get Data
df = pd.read_csv('../input/california-housing-prices/housing.csv')
df.info()
# 
df.describe()
len(df) - df.count()

#from sklearn.model_selection import train_test_split

corr_matrix = df.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
ax = sns.relplot(x="median_income", y="median_house_value", data=df)
plt.figure(figsize=(16,9))

sns.distplot(df["median_income"],label="median income")
 
plt.title("Histogram of Median Income") # for histogram title
plt.legend() # for label
plt.show()
import plotly.express as px
fig = px.box(df, y="median_income")
fig.show()
#https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb
df["income_cat"] = pd.cut(df["median_income"],
                          bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                          labels=[1, 2, 3, 4, 5])
# bar plot i=of income cat
plt.figure(figsize=(12,7))
sns.countplot(x='income_cat', data=df)
 
plt.show()
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)

for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

df = strat_train_set.copy().drop('income_cat', axis=1)

df.info()
df['housing_median_age'] = df['housing_median_age'].astype('uint8')
g = sns.catplot(x="housing_median_age",
                y="median_house_value", 
                data=df, 
                kind="box")
g.fig.set_figwidth(16)
g.fig.set_figheight(9)
import plotly.express as px

fig = px.scatter(df, x="longitude", y="latitude",color="median_house_value")
fig.show()
df['total_rooms'] = df['total_rooms'].astype('uint16')
ax = sns.relplot(x='total_rooms', y="median_house_value", data=df)
sns.catplot(x="ocean_proximity",
                y="median_house_value", 
                data=df, 
                kind="box")
plt.show()
corr_matrix = df.corr()
#corr_matrix["median_house_value"].sort_values(ascending=False)
sns.heatmap(corr_matrix)
corr_matrix["median_house_value"].sort_values(ascending=False)


fig = px.scatter_matrix(df,
    dimensions=["median_house_value", "median_income", "total_rooms", "housing_median_age"])#,
#    color="ocean_proximity")
fig.show()
fig = px.scatter(df, x="median_income",
                 y="median_house_value", 
                 size='total_rooms')
fig.show()
df.info()
from sklearn.impute import KNNImputer


imputer = KNNImputer(n_neighbors=5)
df_filled = imputer.fit_transform(df.drop('ocean_proximity', axis=1))

#lets check the result the 'total_bedrooms' values
np.isnan(df_filled[:,4]).sum()
df['total_bedrooms'] = df_filled[:,4]
pd.get_dummies(df)
from sklearn.base import BaseEstimator, TransformerMixin


class AttributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, num_vars, add_bedrooms_per_room = True): 
        self.add_bedrooms_per_room = add_bedrooms_per_room
        #self._num_vars = num_vars
        self._cols = num_vars
        
    def fit(self, X, y=None):
        return self # nothing else to do
    
    def transform(self, X, y=None):     
        X = pd.DataFrame(X, columns=self._cols)
        X['rooms_per_household'] = X['total_rooms']/X['households']
        X['population_per_household'] = X['population']/X['households']
        #self.new_cols_ = ['rooms_per_household', 'population_per_household']
        if self.add_bedrooms_per_room:
            X['bedrooms_per_room'] = X['total_bedrooms']/X['total_rooms']
            #self.new_cols_.append('bedrooms_per_room')
            
        self._cols = X.columns.tolist()
        return X.values
    
    def get_feature_names(self):
        return self._cols
df.describe().iloc[:, 2:]
df = strat_train_set.copy().drop('income_cat', axis=1)
labels = strat_train_set["median_house_value"].copy()
cat_vars, num_vars = [], []
for var, var_type in df.dtypes.items():
    if var_type =='object':
        cat_vars.append(var)
    else:
        num_vars.append(var)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
#from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline 
from sklearn.compose import ColumnTransformer

#Custom Transformer that extracts columns passed as argument to its constructor 
class FeatureSelector(BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, feature_names):
        self._feature_names = feature_names 
        
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X[self._feature_names].values 
    
#Defining the steps in the categorical pipeline 
cat_pipeline = Pipeline( [ ( 'cat_selector', FeatureSelector(cat_vars) ),
                          ( 'one_hot_encoder', OneHotEncoder( sparse = False ) ) ] )
    
#Defining the steps in the numerical pipeline     
num_pipeline = Pipeline([
        ( 'num_selector', FeatureSelector(num_vars) ),
        ('imputer', KNNImputer(n_neighbors=5)),
        ('attribs_adder', AttributeAdder(num_vars=num_vars, add_bedrooms_per_room = True)),
        ('std_scaler', StandardScaler()),
    ])


#housing_num_tr = num_pipeline.fit_transform(housing_num)

#Combining numerical and categorical piepline into one full big pipeline horizontally 
#using FeatureUnion
full_pipeline = FeatureUnion( transformer_list = [ ( 'num_pipeline', num_pipeline ),
                                                  ( 'cat_pipeline', cat_pipeline )] 
                            )


#num_pipeline.
#num_pipeline.named_steps['attribs_adder'].get_feature_names()
#cat_pipeline.named_steps['one_hot_encoder'].get_feature_names()
df_prepared = full_pipeline.fit_transform(df)
df_prepared
all_vars = (list(full_pipeline.transformer_list[0][1].named_steps['attribs_adder'].get_feature_names()) + 
            list(full_pipeline.transformer_list[1][1].named_steps['one_hot_encoder'].get_feature_names()))
all_vars
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
reg_tree = DecisionTreeRegressor()


scores = cross_val_score(reg_tree, df_prepared, labels,
                         scoring="neg_mean_squared_error",
                         cv=7)

tree_scores = np.sqrt(-scores)

print('CV scores', tree_scores)
print('CV best score', tree_scores.min())
print('CV mean score', tree_scores.mean())
print('CV std score', tree_scores.std())
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(df_prepared, labels)

preds = lin_reg.predict(df_prepared)
lr_rmse = np.sqrt(mean_squared_error(labels, preds))
lr_rmse

scores = cross_val_score(lin_reg, df_prepared, labels,
                         scoring="neg_mean_squared_error",
                         cv=7)

lr_scores = np.sqrt(-scores)
print('CV scores', lr_scores)
print('CV best score', lr_scores.min())
print('CV mean score', lr_scores.mean())
print('CV std score', lr_scores.std())
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=10, max_depth=2)
xgb.fit(df_prepared, labels)

#Now that the model is trained, let’s evaluate it on the training set:


preds = xgb.predict(df_prepared)
xgb_rmse = np.sqrt(mean_squared_error(labels, preds))
xgb_rmse
xgb = XGBRegressor(n_estimators=10, max_depth=2)
scores = cross_val_score(xgb, df_prepared, labels,
                         scoring="neg_mean_squared_error",
                         cv=7)

xgb_scores = np.sqrt(-scores)
print('CV scores', xgb_scores)
print('CV best score', xgb_scores.min())
print('CV mean score', xgb_scores.mean())
print('CV std score', xgb_scores.std())
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(df_prepared, labels)

grid_search.best_params_

np.sqrt(-grid_search.best_score_)
feature_importances = grid_search.best_estimator_.feature_importances_
sorted(zip(feature_importances, all_vars), reverse=True)
#feature_importances

