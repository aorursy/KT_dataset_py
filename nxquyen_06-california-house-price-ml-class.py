# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
csvPath = "/kaggle/input/california-housing-prices/housing.csv"

housing = pd.read_csv(csvPath)

housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()
housing.hist(bins=50, figsize=(20,15))

plt.show()
housing["income_cat"] = pd.cut(housing["median_income"],

                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],

labels=[1, 2, 3, 4, 5])

housing["income_cat"].hist()
trainSet, testSet = train_test_split(housing, test_size=0.2, random_state=42)
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 

for trainIndex, testIndex in split.split(housing, housing["income_cat"]):

        stratTrainSet = housing.loc[trainIndex]

        stratTestSet = housing.loc[testIndex]
testSet["income_cat"].value_counts() / len(stratTestSet)
stratTestSet["income_cat"].value_counts() / len(stratTestSet)
housing["income_cat"].value_counts() / len(housing)
## Put the test set aside and only explore the training set.

housing = stratTrainSet.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
housing.plot(

    kind="scatter", x="longitude", y="latitude", alpha=0.4,

    s=housing["population"]/100, label="population", figsize=(10,7),

    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,

)

plt.legend()
corrMatrix = housing.corr()
corrMatrix["median_house_value"].sort_values(ascending=False)
attributes = ["median_house_value", "median_income", "total_rooms",

              "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12, 8))
## The most promising attribute to predict the median house value is the median income

housing.plot(kind="scatter", x="median_income", y="median_house_value",

                 alpha=0.1)
## Try out various attribute combinations

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]

housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]

housing["population_per_household"]=housing["population"]/housing["households"]
corrMatrix = housing.corr()

corrMatrix["median_house_value"].sort_values(ascending=False)
housing = stratTrainSet.drop("median_house_value", axis=1)

housingLabels = stratTrainSet["median_house_value"].copy()
housing.dropna(subset=["total_bedrooms"]) # option 1 

housing.drop("total_bedrooms", axis=1) # option 2 

median = housing["total_bedrooms"].median() # option 3 

housing["total_bedrooms"].fillna(median, inplace=True)
from sklearn.impute import SimpleImputer 

imputer = SimpleImputer(strategy="median")
## The median can only be computed on numerical attributes, 

## we need to create a copy of the data without the text attribute ocean_proximity:

housingNum = housing.drop("ocean_proximity", axis=1)
imputer.fit(housingNum)
housingNum.median().values
## Use this imputer to transform the training set by replacing missing values by the learned medians

X = imputer.transform(housingNum)
housingCat = housing[["ocean_proximity"]]

housingCat.head(10)
# Convert these categories from text to numbers

from sklearn.preprocessing import OrdinalEncoder

ordinalEncoder = OrdinalEncoder()
housingCatEncoded = ordinalEncoder.fit_transform(housingCat)

housingCatEncoded[:10]
ordinalEncoder.categories_
from sklearn.preprocessing import OneHotEncoder

catEncoder = OneHotEncoder()

housingCatOnehot = catEncoder.fit_transform(housingCat)

housingCatOnehot

#  the output is a SciPy sparse matrix, instead of a NumPy array.
catEncoder.categories_
from sklearn.base import BaseEstimator, TransformerMixin



rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs

        self.add_bedrooms_per_room = add_bedrooms_per_room 

    def fit(self, X, y=None):

        return self # nothing else to do 

    def transform(self, X, y=None):

        rooms_per_household = X[:, rooms_ix] / X[:, households_ix] 

        population_per_household = X[:, population_ix] / X[:, households_ix] 

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

            return np.c_[X, rooms_per_household, population_per_household,

            bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household, population_per_household]



attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

housing_extra_attribs = attr_adder.transform(housing.values)
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

numPipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('attribs_adder', CombinedAttributesAdder()),

        ('std_scaler', StandardScaler()),

    ])

housingNumTr = numPipeline.fit_transform(housingNum)
from sklearn.compose import ColumnTransformer 

numAttribs = list(housing_num)

catAttribs = ["ocean_proximity"]

fullPipeline = ColumnTransformer([

     ("num", numPipeline, numAttribs),

     ("cat", OneHotEncoder(), catAttribs),

 ])

housingPrepared = fullPipeline.fit_transform(housing)
from sklearn.linear_model import LinearRegression 

linReg = LinearRegression()

linReg.fit(housingPrepared, housingLabels)
someData = housing.iloc[:5]

someLabels = housingLabels.iloc[:5]

someDataPrepared = fullPipeline.transform(someData)

print("Predictions:", lin_reg.predict(someDataPrepared))
from sklearn.metrics import mean_squared_error

housingPreedictions = linReg.predict(housingPreepared)

linMse = mean_squared_error(housingLabels, housingPreedictions) 

linRmse = np.sqrt(linMse)

linRmse
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBoostClassifier
from sklearn.model_selection import GridSearchCV

paramGrid = [

    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},

]

forestReg = RandomForestRegressor()

gridSearch = GridSearchCV(forestReg, paramGrid, cv=5,

                           scoring='neg_mean_squared_error',

                           return_train_score=True)

gridSearch.fit(housingPrepared, housingLabels)
gridSearch.best_params_ 

gridSearch.best_estimator_
cvres = gridSearch.cv_results_

for meanScore, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-meanScore), params)
feature_importances = grid_search.best_estimator_.feature_importances_ 

feature_importances
finalModel = gridSearch.best_estimator_

XTest = stratTestSet.drop("median_house_value", axis=1)

yTest = stratTestSet["median_house_value"].copy()

XTestPrepared = fullPipeline.transform(XTest)

finalPredictions = finalModel.predict(XTestPrepared)

finalMse = mean_squared_error(yTest, finalPredictions) 

finalRmse = np.sqrt(finalMse)
from scipy import stats

confidence = 0.95

squared_errors = (final_predictions - y_test) ** 2

np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,

                         loc=squared_errors.mean(), 

                         scale=stats.sem(squared_errors)))