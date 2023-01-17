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
df = pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")
df.info()
display(df.isna().sum())
import matplotlib.pyplot as plt 
df.hist(bins=50, figsize=(20,15)) 
plt.show()
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

print("train : ",len(train_set))
print("test : ",len(test_set))
df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.3,    
             s=df["population"]/100, label="population", figsize=(10,7),    
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True) 
plt.legend()
import seaborn as sns

correlation = train_set.corr()

fig = plt.subplots(figsize=(10,10))
sns.heatmap(correlation,vmax=1,square=True,annot=True,cmap='Blues')
from pandas.plotting import scatter_matrix

features = ['median_income', 'median_house_value']

scatter_matrix(train_set[features], figsize=(6,6))
train_data = train_set.drop('median_house_value', axis=1)
train_label = train_set['median_house_value'].copy()

test_data = test_set.drop('median_house_value', axis=1)
test_label = test_set['median_house_value'].copy()

print('labels created')
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

train_data_nums = train_data.drop("ocean_proximity", axis=1)
from sklearn.base import BaseEstimator, TransformerMixin

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

num_attribs = list(train_data_nums) 
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

train_prepared = full_pipeline.fit_transform(train_data)
test_prepared = full_pipeline.fit_transform(test_data)
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
# linear_model.fit(train_prepared, test_label)

# linear_predictions = tree_reg.predict(test_prepared)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(linear_model, train_prepared, train_label, scoring="neg_mean_squared_error", cv=10) 
linear_rmse_scores = np.sqrt(-scores)

print("Mean:", linear_rmse_scores.mean())
print("Standard deviation:", linear_rmse_scores.std())
from sklearn.tree import DecisionTreeRegressor

tree_model = DecisionTreeRegressor()
# tree_model.fit(test_prepared, test_label)
from sklearn.model_selection import cross_val_score 

scores = cross_val_score(tree_model, train_prepared, train_label, scoring="neg_mean_squared_error", cv=10) 
tree_rmse_scores = np.sqrt(-scores)

print("Mean:", tree_rmse_scores.mean())
print("Standard deviation:", tree_rmse_scores.std())
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor()
from sklearn.model_selection import cross_val_score 

scores = cross_val_score(forest_model, train_prepared, train_label, scoring="neg_mean_squared_error", cv=10) 
forest_rmse_scores = np.sqrt(-scores)

print("Mean:", forest_rmse_scores.mean())
print("Standard deviation:", forest_rmse_scores.std())