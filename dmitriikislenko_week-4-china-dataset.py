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
houses = pd.read_csv('/kaggle/input/lianjia/new.csv',  encoding= 'unicode_escape', low_memory=False)
houses.head()


houses.info()
houses['constructionTime'] = houses['constructionTime'].astype('str') 
condition = (houses.constructionTime == '0') | (houses.constructionTime == 'Î´Öª')
column_name = 'constructionTime'
houses.loc[condition, column_name] = '2018'

condition = (houses.constructionTime == '1')
column_name = 'constructionTime'
houses.loc[condition, column_name] = '2017'
houses['year'] = pd.to_datetime(houses['constructionTime'])
houses['houseAge'] = (pd.datetime.today() - houses['year']).dt.days*0.00273973
houses.describe()
%matplotlib inline
import matplotlib.pyplot as plt
houses.hist(bins=50, figsize=(20,15))
# save_fig("attribute_histogram_plots")
plt.show()
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(houses, test_size=0.2, random_state=42)
houses = train_set.copy()
houses.plot(kind="scatter", x="Lng", y="Lat", alpha=0.3,
    s=houses["communityAverage"]/10000, label="Community average", figsize=(10,7),
    c="price", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()
corr_matrix = houses.corr()
corr_matrix["price"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix

attributes = ["price", "communityAverage", "DOM",
              "renovationCondition"]
scatter_matrix(houses[attributes], figsize=(12, 8))
from sklearn.impute import SimpleImputer

houses_labels = train_set["price"].copy()
imputer = SimpleImputer(strategy="median")
houses_num = houses.drop(["url", 'id', 'tradeTime', 'livingRoom', 'drawingRoom', 'bathRoom', 'floor', 'constructionTime', 'year'], axis=1)
imputer.fit(houses_num)
test = imputer.transform(houses_num)
houses_trained = pd.DataFrame(test, columns=houses_num.columns,
                          index=houses_num.index)
houses_trained
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
houses_categories = houses[["floor"]]
housing_cat_encoded = ordinal_encoder.fit_transform(houses_categories)
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(houses_categories)
housing_cat_1hot
housing_cat_1hot.toarray()

from sklearn.base import BaseEstimator, TransformerMixin

price_ix, communityAverage_ix = 6, 18

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_pricePerCommunityAverage=True): # no *args or **kargs
        self.add_pricePerCommunityAverage = add_pricePerCommunityAverage
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        if (X[:, communityAverage_ix] == 0).any():
            pricePerCommunityAverage = X[:, price_ix] / 0.001
        else:
            pricePerCommunityAverage = X[:, price_ix] / X[:, communityAverage_ix]
        if self.add_pricePerCommunityAverage:
            return np.c_[X, pricePerCommunityAverage]

attr_adder = CombinedAttributesAdder(add_pricePerCommunityAverage=False)
houses_extra_attribs = attr_adder.transform(houses.values)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

houses_num_tr = num_pipeline.fit_transform(houses_num)
from sklearn.compose import ColumnTransformer

num_attribs = list(houses_num)

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
    ])

houses_prepared = full_pipeline.fit_transform(houses)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(houses_prepared, houses_labels)
some_data = houses.iloc[:10]
some_labels = houses_labels.iloc[:10]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))

print("Labels:", list(some_labels))
from sklearn.metrics import mean_squared_error
houses_predictions = lin_reg.predict(houses_prepared)
lin_mse = mean_squared_error(houses_labels, houses_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(houses_prepared, houses_labels)
housing_predictions = tree_reg.predict(houses_prepared)
tree_mse = mean_squared_error(houses_labels, houses_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, houses_prepared, houses_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_scores(tree_rmse_scores)
lin_scores = cross_val_score(lin_reg, houses_prepared, houses_labels,
                              scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = [
    {'n_estimators': [10, 20], 'max_features': [1, 2]},
  ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=2,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(houses_prepared, houses_labels)
grid_search.best_params_
final_model = lin_reg

X_test = test_set
y_test = test_set["price"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse