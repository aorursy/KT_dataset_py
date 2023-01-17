import pandas as pd

import numpy as np

from subprocess import check_output

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import FeatureUnion

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

from sklearn.svm import SVR

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import reciprocal, uniform
class CatFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        Z = pd.get_dummies(X[self.attribute_names], drop_first = False)

        return Z.values



## From Ref[1]

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values

    

## Ref[4] & Ref[3]

class BetterImputer: 

    def __init__(self, columns = None):

        self.columns = columns

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        output = X.copy()

        for col in self.columns:

            output[col].fillna = output[col].value_counts().index[0]

        return output

    

    def fit_transform(self, X, y=None):

        return self.fit(X,y).transform(X)
## Feature engineering - from tutorial by juliencs

## Some numerical features are actually really categories 

def num_to_cat(X):

    X = X.replace({#"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 

                   #                    50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 

                   #                    80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 

                   #                    150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},

                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",

                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}

                      })

    return(X)



## Change categorical features into ordered numbers

def cat_to_num(X):

    X = X.replace({"ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 

                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},

                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},

                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},

                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

                       "Street" : {"Grvl" : 1, "Pave" : 2},

                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}

                  })

    return(X)



## Simplify existing features

def simplify_features(X):

    X["SimplOverallQual"] = X.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad

                                                       4 : 2, 5 : 2, 6 : 2, # average

                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good

                                                      })

    X["SimplOverallCond"] = X.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, # bad

                                                       4 : 2, 5 : 2, 6 : 2, # average

                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good

                                                      })

    X["SimplFunctional"] = X.Functional.replace({1 : 1, 2 : 1, # bad

                                                     3 : 2, 4 : 2, # major

                                                     5 : 3, 6 : 3, 7 : 3, # minor

                                                     8 : 4 # typical

                                                    })

    X["SimplKitchenQual"] = X.KitchenQual.replace({1 : 1, # bad

                                                       2 : 1, 3 : 1, # average

                                                       4 : 2, 5 : 2 # good

                                                      })

    X["SimplHeatingQC"] = X.HeatingQC.replace({1 : 1, # bad

                                                   2 : 1, 3 : 1, # average

                                                   4 : 2, 5 : 2 # good

                                                  })

    X["SimplExterCond"] = X.ExterCond.replace({1 : 1, # bad

                                                   2 : 1, 3 : 1, # average

                                                   4 : 2, 5 : 2 # good

                                                  })

    X["SimplExterQual"] = X.ExterQual.replace({1 : 1, # bad

                                                   2 : 1, 3 : 1, # average

                                                   4 : 2, 5 : 2 # good

                                                  })

    return(X)



def create_new_features(X):

    # Overall quality of the house

    X["OverallGrade"] = X["OverallQual"] * X["OverallCond"]



    # Overall quality of the exterior

    X["ExterGrade"] = X["ExterQual"] * X["ExterCond"]



    # Overall kitchen score

    X["KitchenScore"] = X["KitchenAbvGr"] * X["KitchenQual"]



    # Simplified overall quality of the house

    X["SimplOverallGrade"] = X["SimplOverallQual"] * X["SimplOverallCond"]



    # Simplified overall quality of the exterior

    X["SimplExterGrade"] = X["SimplExterQual"] * X["SimplExterCond"]



    # Simplified overall kitchen score

    X["SimplKitchenScore"] = X["KitchenAbvGr"] * X["SimplKitchenQual"]



    # Total number of bathrooms

    X["TotalBath"] = X["BsmtFullBath"] + (0.5 * X["BsmtHalfBath"]) + X["FullBath"] + (0.5 * train["HalfBath"])



    # Total SF for house (incl. basement)

    X["AllSF"] = X["GrLivArea"] + X["TotalBsmtSF"]



    # Total SF for 1st + 2nd floors

    X["AllFlrsSF"] = X["1stFlrSF"] + X["2ndFlrSF"]



    # Total SF for porch

    X["AllPorchSF"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]



    # House completed before sale or not

    X["BoughtOffPlan"] = X.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 

                                                      "Family" : 0, "Normal" : 0, "Partial" : 1})

    return(X)



## https://pandas.pydata.org/pandas-docs/stable/categorical.html

def remove_categories(df1, df2, cat_cols):

    for i in cat_cols:

        a = df1[i].astype('category')

        a = a.cat.categories

    

        b = df2[i].astype('category')

        b = b.cat.categories

    

        ## That which is in train which is not in test

        d = list(set(a) - set(b))

    

        ## Therefore these need to be removed from train

        for j in d:

            df1.drop(df1[df1[i] == j].index, inplace = True)

            print("{} removed from {} in df1".format(j,i))

            

    return(df1)
## Load data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
## Why am I dropping Eletrical again?

# train.drop(train.loc[train['Electrical'].isnull()].index)

# test.drop(test.loc[test['Electrical'].isnull()].index)



## Create submission dataframe and drop ids from train and test

submission = pd.DataFrame()

submission['Id'] = test['Id']

test.drop(['Id'], axis = 1, inplace=True)

train.drop(['Id'], axis = 1, inplace = True)
test_num_columns = test._get_numeric_data().columns

train_num_columns = train._get_numeric_data().columns

test_cat_columns = list(set(test.columns) - set(test_num_columns))

train_cat_columns = list(set(train.columns) - set(train_num_columns))
## Create list of numerical and categorical attributes

num_cols = test._get_numeric_data().columns

cat_cols = list(set(test.columns) - set(num_cols))

num_cols = list(num_cols)



## Columns in test which are not in train should be removed          

          

## Copy that which is in test (and which is not in train) to train and impute the median response - rows

train = remove_categories(train, test, cat_cols)



## Create independent submission and price vectors

price_labels = np.log(train["SalePrice"].copy())

train.drop(["SalePrice"], axis=1, inplace = True)



## Make sure that the training data only reflects the columns in the testing data 

names_in_test = list(test)

train = train[names_in_test]
## missing data Ref[3] - these should really be functions, too

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)



## remove columns from train

train = train.drop((missing_data[missing_data['Total'] > 1]).index, 1)



# remove columns from test Ref[3]

test = test.drop((missing_data[missing_data['Total'] > 1]).index,1)



# just checking that there's no missing data missing...

# train.isnull().sum().max()

# test.isnull().sum().max()
## Run training data through feature functions - need to add them to the pipeline

train_data = num_to_cat(train)

train_data = cat_to_num(train_data)

train_data = simplify_features(train_data)

train_data = create_new_features(train_data)



## Run training data through feature functions - need to add them to the pipeline

test_data = num_to_cat(test)

test_data = cat_to_num(test_data)

test_data = simplify_features(test_data)

test_data = create_new_features(test_data)
## Find missing data in Test set and impute the MODE of the column

total = test_data.isnull().sum().sort_values(ascending=False)

percent = (test_data.isnull().sum()/test_data.isnull().count()*100).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



def impute_mode(X, col_name):

    fixed_col = X[col_name].replace(np.nan, X[col_name].value_counts().index[0], regex=True)

    return fixed_col



for col_name in list(missing_data.index):

    test_data[col_name] = impute_mode(test_data, col_name)
## Create list of numerical and categorical attributes

num_cols = test_data._get_numeric_data().columns

cat_cols = list(set(test_data.columns) - set(num_cols))

num_cols = list(num_cols)
# numerical pipeline

num_pipeline = Pipeline([

    ('num_selector', DataFrameSelector(num_cols)),

    ('std_scaler', StandardScaler()),

])



cat_pipeline = Pipeline([

    ('cat_selector', CatFrameSelector(cat_cols)),

])



# unite

full_pipeline = FeatureUnion(transformer_list = [

    ('num_pipeline', num_pipeline),

    ('cat_pipeline', cat_pipeline),

])
# run pipeline on training data

train_prepared = full_pipeline.fit_transform(train_data)



# run pipeline on testing data

test_prepared = full_pipeline.transform(test_data)
## define function to plot learning curves

def plot_learning_curves(model, X, y):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

    train_errors, val_errors = [], []

    for m in range(1, len(X_train)):

        model.fit(X_train[:m], y_train[:m])

        y_train_predict = model.predict(X_train[:m])

        y_val_predict = model.predict(X_val)

        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))

        val_errors.append(mean_squared_error(y_val_predict, y_val))

    plt.plot(np.sqrt(train_errors), "r-", linewidth=2, label="train")

    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")

    plt.ylim(0,1)

    plt.legend(loc="upper right", fontsize=14)

    plt.xlabel("Training set size", fontsize=14)

    plt.ylabel("RMSE(log(y))", fontsize=14)



## Run

plot_learning_curves(SVR(), train_prepared, price_labels)
param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}

rnd_search_cv = RandomizedSearchCV(SVR(), param_distributions, n_iter=10, verbose=2, random_state=42)

rnd_search_cv.fit(train_prepared, price_labels)
rnd_search_cv.best_estimator_
rnd_search_cv.best_params_
y_pred = rnd_search_cv.best_estimator_.predict(train_prepared)

mse = mean_squared_error(price_labels, y_pred)

np.exp(np.sqrt(mse))
predictions = np.exp(rnd_search_cv.best_estimator_.predict(test_prepared))

submission['SalePrice'] = pd.Series(predictions, index = submission.index)

submission.to_csv('submission.csv',index=False)
submission
print(check_output(["ls", "./"]).decode("utf8"))