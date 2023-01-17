# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/housing.csv')

df.head(5)
aaa = df[['total_rooms', 'population']]

aaa.head(3)
df.info()
df.describe()
df["ocean_proximity"].value_counts()
%matplotlib inline

import matplotlib.pyplot as plt

df.hist(bins=50, figsize=(20,15))
np.random.seed(42)

def split_train_test(data, test_ratio):

    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]

    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]



train_set, test_set = split_train_test(df, 0.2)

print(len(train_set))

print(len(test_set))
df.plot(kind="scatter", x="longitude", y="latitude")
df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.05)
df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

    s=df["population"]/100, label="population", figsize=(10,7),

    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,

)

plt.legend()
corr_matrix = df.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)
import seaborn as sns

attrs = ["median_house_value", "median_income", "total_rooms","housing_median_age"]

sns.pairplot(data=df[attrs], height=2, aspect=1.5)
df.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
df["rooms_per_household"] = df["total_rooms"]/df["households"]

df["bedrooms_per_room"] = df["total_bedrooms"]/df["total_rooms"]

df["population_per_household"]=df["population"]/df["households"]

corr_matrix = df.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2, random_state=21)

print(len(train), len(test))

train.describe()
X_train = train.drop('median_house_value', axis=1)

y_train = train['median_house_value'].copy()

X_train.describe()
y_train.describe()
option1 = df.dropna(subset=['total_bedrooms'])

option2 = df.drop('total_bedrooms', axis=1)

median_num_bedrooms = df['total_bedrooms'].median()

df['total_bedrooms'].fillna(median_num_bedrooms, inplace=True)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

df_num = df.drop("ocean_proximity", axis=1)

imputer.fit(df_num)

print(imputer.statistics_)

print(df_num.median().values)
X = imputer.transform(df_num)

# X is a raw numpy array, turn it back into a dataframe

X = pd.DataFrame(X, columns=df_num.columns)

X.describe()

df_cat = df[['ocean_proximity']]

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()

df_cat_encoded = ordinal_encoder.fit_transform(df_cat)

# go back to a dataframe from the raw NumPy array outputted 

df_cat_encoded = pd.DataFrame(df_cat_encoded, columns=df_cat.columns)

print(df_cat_encoded["ocean_proximity"].value_counts())

print(ordinal_encoder.categories_)
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()

df_cat_1hot = cat_encoder.fit_transform(df_cat)

df_cat_1hot
print(df_cat_1hot.toarray())
from sklearn.compose import ColumnTransformer

df1 = train.drop('median_house_value', axis=1)

num_attrs = list(df1)

num_attrs.remove("ocean_proximity")

cat_attrs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([

        ("num", SimpleImputer(strategy='median'),num_attrs),

        ("cat", OneHotEncoder(), cat_attrs),

    ])

X = full_pipeline.fit_transform(df1)

print(X)
y = train['median_house_value'].values

print(y)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X, y)
print(lin_reg.predict( X[:5]))
print(y[:5])
from sklearn.metrics import mean_squared_error

preds = lin_reg.predict(X)

mse = mean_squared_error(y, preds)

print(np.sqrt(mse))
sns.scatterplot(y, preds)
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(X, y)
tree_preds = tree_reg.predict(X)

tree_mse = mean_squared_error(y, tree_preds)

print(np.sqrt(tree_mse))
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, X, y,

                         scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)

print(tree_rmse_scores)

print(tree_rmse_scores.mean())

print(tree_rmse_scores.std())
scores = cross_val_score(lin_reg, X, y,

                         scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-scores)

print(lin_rmse_scores)

print(lin_rmse_scores.mean())

print(lin_rmse_scores.std())
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=30)

scores = cross_val_score(forest_reg, X, y,

                         scoring="neg_mean_squared_error", cv=10)

rf_rmse_scores = np.sqrt(-scores)

print(rf_rmse_scores)

print(rf_rmse_scores.mean())

print(rf_rmse_scores.std())
rf_tupels = [[f, 'rf', x] for f,x in zip(range(10), rf_rmse_scores)]

dt_tupels = [[f, 'dt', x] for f,x in zip(range(10), tree_rmse_scores)]

lr_tupels = [[f, 'lr', x] for f,x in zip(range(10), lin_rmse_scores)]

results = pd.DataFrame(rf_tupels+dt_tupels+lr_tupels, columns=['fold', 'algo', 'rmse'])



sns.scatterplot(data=results, x='fold', y='rmse', hue='algo')

test.describe()
df2 = test.drop('median_house_value', axis=1)

X_test = full_pipeline.transform(df2)

y_test = test['median_house_value'].values

pred_test = lin_reg.predict(X_test)

mse_test = mean_squared_error(y_test, pred_test)

print(np.sqrt(mse_test))

forest_reg.fit(X, y)

pred_test_rf = forest_reg.predict(X_test)

mse_test_rf = mean_squared_error(y_test, pred_test_rf)

print(np.sqrt(mse_test_rf))
sns.scatterplot(y_test, pred_test)
sns.scatterplot(y_test, pred_test_rf)
from sklearn.base import BaseEstimator



class BoundedClassifier(BaseEstimator):

    def __init__(self, learner):

        self.learner = learner

    def fit(self, X, y):

        self.learner.fit(X, y)

        self.min_y = np.min(y)

        self.max_y = np.max(y)

        self.mean_y = np.mean(y)

        self.lo = np.percentile(y, 5)

        self.hi = np.percentile(y, 95)

        print(self.lo, self.hi)

    def predict(self, X):

        preds = self.learner.predict(X)

        #preds[ preds < self.min_y ] = self.mean_y

        #preds[ preds > self.max_y ] = self.mean_y

        #preds[ preds < self.min_y ] = self.min_y

        #preds[ preds > self.max_y ] = self.max_y

        preds[ preds < self.min_y ] = self.lo

        preds[ preds > self.max_y ] = self.hi        

        return preds

    

bounded_model = BoundedClassifier(LinearRegression())

print(bounded_model)
bounded_model.fit(X, y)

pred_test_bm = bounded_model.predict(X_test)

mse_test_bm = mean_squared_error(y_test, pred_test_bm)

print(bounded_model.min_y, bounded_model.max_y, np.sqrt(mse_test_bm))
sns.scatterplot(y_test, pred_test_bm)