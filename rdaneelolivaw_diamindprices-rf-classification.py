# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)s



import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



data_raw = pd.read_csv('../input/diamonds.csv')

data = pd.read_csv('../input/diamonds.csv')



data_raw.drop(['Unnamed: 0'], axis=1, inplace=True)

data.drop(['Unnamed: 0'], axis=1, inplace=True)



data_num = data.select_dtypes(exclude=[object]).columns
data_raw.head()
data_raw.tail()
data_raw.shape
data_raw.columns
data_raw.info()
data_raw.describe()
sns.distplot(data_raw['carat'], color="b", rug=True)
sns.distplot(data_raw[data_raw['carat'] < 2].carat, color="b", rug=True)
sns.distplot(data_raw['depth'], color="b", rug=True)
sns.distplot(data_raw['table'], color="b", rug=True)
sns.distplot(data_raw['price'], color="b", rug=True)
sns.scatterplot(x="carat", y="price", 

                hue="depth",

                sizes=(1, 8), linewidth=0,

                data=data_raw)
data_raw['cut'].value_counts()
data_raw['color'].value_counts()
data_raw['clarity'].value_counts()
data_raw.hist(bins=50, figsize=(20,15))
# processing pice feature

price_cat = np.ceil(data['price']/250)

data['price_cat'] = price_cat

sns.distplot(data['price_cat'], color="b", rug=True)
#test train set



from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(data, data['price_cat']):

    strat_train_set = data.loc[train_index]

    strat_test_set = data.loc[test_index]

strat_train_set.drop('price_cat', axis=1, inplace=True)

strat_train_set.shape
strat_test_set.drop('price_cat', axis=1, inplace=True)

strat_test_set.shape
from sklearn.base import BaseEstimator, TransformerMixin



class PriceCatgorization(BaseEstimator, TransformerMixin):

    def __init__(self, price_denominator):

        self.price_denominator = price_denominator

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        price_cat = np.ceil(X['price']/self.price_denominator)

        X['price_cat'] = price_cat

        return X

        
class CutColorClarityCatgorization(BaseEstimator, TransformerMixin):

    def __init__(self):

        return

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        cut_dummies = pd.get_dummies(X['cut'] ,prefix='cut', drop_first=True)

        color_dummies = pd.get_dummies(X['color'] ,prefix='color', drop_first=True)

        clarity_dummies = pd.get_dummies(X['clarity'] ,prefix='clarity', drop_first=True)

        

        X = pd.concat([X, cut_dummies, color_dummies, clarity_dummies], axis=1)

        X.drop(['price', 'cut', 'color', 'clarity'], axis=1, inplace=True)

        

        return X
# NOT NEEDED ANYMORE USING COLUMN_TRANSFORMER POST SKLEARN 2.0+



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler



num_attr = ["carat", "depth", "table", "x", "y", "z"]

cat_attr = ["price", "cut", "color", "clarity"]



num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy="median")),

    ('std_scaler', StandardScaler())

])



cat_pipeline = Pipeline([

    ('pice_cat', PriceCatgorization(250)),

    ('cccc', CutColorClarityCatgorization())

])
from sklearn.compose import ColumnTransformer



full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attr),

        ("cat", cat_pipeline, cat_attr),

    ])
data_training_prepared = full_pipeline.fit_transform(strat_train_set)

data_training_prepared.shape
data_test_prepared = full_pipeline.fit_transform(strat_test_set)

data_test_prepared.shape
# extract X and y vectors

y_train = data_training_prepared[:, [6]]

print(y_train[5000])

X_train = np.delete(data_training_prepared, 6, 1)

print(X_train[5000])
# LINEAR REGRESSION



from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)
# LINER REGRESSION RESULTS

# Cross validation



def display_scores(score):

    print("Scores:",score)

    print("Mean:", score.mean())

    print("Standard dviation:", score.std())



from sklearn.model_selection import cross_val_score



lin_reg_score = cross_val_score(lin_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10)

lin_reg_rmse_score = np.sqrt(-lin_reg_score)



display_scores(lin_reg_rmse_score)
# RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor()

forest_reg.fit(X_train, y_train)



forest_reg_score = cross_val_score(forest_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10)

forest_reg_rmse_score = np.sqrt(-forest_reg_score)



display_scores(forest_reg_rmse_score)

from sklearn.metrics import mean_squared_error



predictions_training = forest_reg.predict(X_train)



train_mse = mean_squared_error(y_train, predictions_training)

train_rmse = np.sqrt(train_mse)



print("Training RMSE :", train_rmse)
# extract X and y vectors

y_test = data_test_prepared[:, [6]]

# print(y_test[5000])

X_test = np.delete(data_test_prepared, 6, 1)

# print(X_test[5000])



predictions_test_set = forest_reg.predict(X_test)



# from sklearn.metrics import mean_squared_error



final_mse = mean_squared_error(y_test, predictions_test_set)

final_rmse = np.sqrt(final_mse)



print("Final RMSE : ",final_rmse)