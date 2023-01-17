import matplotlib 

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import sklearn

import sklearn.linear_model as linear_model
import os

datapath = os.path.join("../input/house-prices-advanced-regression-techniques/")

housing = pd.read_csv(datapath + "train.csv");

testdf = pd.read_csv(datapath + "test.csv");

sampleSubmissiondf = pd.read_csv(datapath + "sample_submission.csv");
pd.set_option('display.max_columns', None)  



print(housing.head())

print(housing.describe())

print(housing.info())
from sklearn.ensemble import RandomForestRegressor





# Create a copy to work with

X = housing.copy()



# Save and drop labels

y = housing.SalePrice

X = X.drop('SalePrice', axis=1)



# fill NANs

X = X.fillna(-999)



# Label Encoder

for c in housing.columns[housing.dtypes == 'object']:

  X[c] = X[c].factorize()[0]



rf = RandomForestRegressor()

rf.fit(X,y)
rf.feature_importances_
plt.figure(figsize=(20,10))

plt.grid(True)

plt.plot(rf.feature_importances_)

plt.xticks(np.arange(X.shape[1]), X.columns.tolist(), rotation=90)
%matplotlib inline

import matplotlib.pyplot as plt

X.hist(bins=50, figsize=(20,15))

plt.show()
column_names_low = ["Functional", "Alley", "Fence", "ExterCond", "PoolQC","Utilities","MiscFeature","Id","SalePrice","MSSubClass"]



column_names_ordinal = ["ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC","KitchenQual", "FireplaceQu","GarageQual", "GarageCond","PoolQC"]

column_names_high = ["OverallQual", "GrLivArea","MasVnrArea","YearBuilt","YearRemodAdd","Neighborhood","LotArea","LotFrontage" ]

conv_dict={'NA':1.,'Po':2.,'Fa':3.,'TA':4.,'Gd':5.,'Ex':6.,np.nan:1.}



df_ord_to_num=housing[column_names_ordinal].replace(conv_dict)

print(df_ord_to_num.head())
# explore test data

# fill NANs

test = testdf.fillna(-999)



# Label Encoder

for c in testdf.columns[testdf.dtypes == 'object']:

  test[c] = test[c].factorize()[0]



%matplotlib inline

import matplotlib.pyplot as plt

test.hist(bins=50, figsize=(20,15))

plt.show()
from sklearn.model_selection import cross_val_score



scores_rf = cross_val_score(rf, X, y, scoring="neg_mean_squared_error", cv=10)

print(scores_rf)
import numpy as np



np.random.seed(42)



def split_train_test(data, test_ratio):

  shuffled_indices = np.random.permutation(len(data))

  test_set_size = int(len(data)*test_ratio)

  test_indices = shuffled_indices[:test_set_size]

  train_indices = shuffled_indices[test_set_size:]

  print(shuffled_indices)

  print(test_indices)

  print(train_indices)

  return data.iloc[train_indices], data.iloc[test_indices]
train_set, test_set = split_train_test(housing,0.2)

print(len(train_set), "train + ", len(test_set), "test")
from sklearn.model_selection import train_test_split



train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


train_features = train_set.drop("SalePrice", axis=1)

train_labels = train_set["SalePrice"].copy()



test_features = test_set.drop("SalePrice", axis=1)

test_labels = test_set["SalePrice"].copy()
train_set.head()
from sklearn.base import BaseEstimator

from sklearn.base import TransformerMixin

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer 



class ReduceCategoriesEncoder(BaseEstimator, TransformerMixin):



    def __init__(self, threshold=0):

        self.threshold = threshold



    def transform(self, X):

        output = X.copy()

        if self.threshold != 0:

            counts = output.count()

            for column in output.columns:

                output[column] =  output[column].apply(str)

                counts= output[column].value_counts()

                repl = counts[counts <= self.threshold].index

                print(repl)

                output[column].replace(repl, 'uncommon',inplace=True)

        return output



    def fit(self, X, y=None):

        return self




class ColumnAdder(BaseEstimator, TransformerMixin):



    def transform(self, X):

        output = X.copy()

        output['SquareFtSum'] = 0

        for column in output.columns:

            output[column].replace(np.nan,0,inplace=True)

            output['SquareFtSum'] =  output[column] + output['SquareFtSum']

        return pd.DataFrame(output['SquareFtSum'])



    def fit(self, X, y=None):

        return self




class CustomEncoder(BaseEstimator, TransformerMixin):



    def __init__(self, dictionary=None):

        self.dictionary = dictionary



    def transform(self, X):

        output = X.copy()

        if self.dictionary != None:

            output.replace(self.dictionary, inplace=True)

        return output



    def fit(self, X, y=None):

        return self
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import FeatureUnion

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestRegressor





cat_features=list(set([i for i in train_features.columns if train_features.dtypes[i]=='object']) - set(column_names_low) -set(column_names_ordinal))

cat_features.append("MSSubClass")



squareFeet_features=["BsmtFinSF1","BsmtFinSF2","1stFlrSF","2ndFlrSF","LowQualFinSF","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MasVnrArea","LotArea"]

num_features = list(set(train_features._get_numeric_data().columns) - set(column_names_low))

ordinal_features = list(set(column_names_ordinal)-set(column_names_low))

# cat_features = ["Neighborhood"]





squareFeetCombiner = Pipeline([

    ('combiner', ColumnAdder()),

    ('std_scaler', StandardScaler())

])



numeric_transformer = Pipeline([

    ('imputer', SimpleImputer (strategy='median')),

    ('std_scaler', StandardScaler())

])



categorical_transformer = Pipeline([

    ('ReduceCategoriesEncoder',ReduceCategoriesEncoder(threshold=100)),

     ('imputer', SimpleImputer (strategy='most_frequent')),

    ('labelBinarizer',OneHotEncoder(sparse=False,handle_unknown='ignore'))

])



ordinal_transformer = Pipeline([

    ('replacer',CustomEncoder(dictionary=conv_dict)),

    ('std_scaler', StandardScaler())

])

preprocessor = ColumnTransformer(

    transformers=[

        ('squareFeetCombiner',squareFeetCombiner,squareFeet_features),

        ('num', numeric_transformer, num_features),

        ('cat', categorical_transformer, cat_features),

        ('ordinal', ordinal_transformer, ordinal_features)

    ])



fullpipeline = Pipeline([

    ("preprocessor",preprocessor),

])


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state = 42)

from pprint import pprint

# Look at parameters used by our current forest

print('Parameters currently in use:\n')

pprint(rf.get_params())





from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

pprint(random_grid)

{'bootstrap': [True, False],

 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],

 'max_features': ['auto', 'sqrt'],

 'min_samples_leaf': [1, 2, 4],

 'min_samples_split': [2, 5, 10],

 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
train_features_transformed = fullpipeline.fit_transform(train_features)



# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(train_features_transformed, train_labels)
rf_random.best_estimator_

train_features_transformed = fullpipeline.fit_transform(train_features)

def evaluate(model, test_features, test_labels):

    predictions = model.predict(test_features)

    errors = abs(predictions - test_labels)

    mape = 100 * np.mean(errors / test_labels)

    accuracy = 100 - mape

    print('Model Performance')

    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))

    print('Accuracy = {:0.2f}%.'.format(accuracy))

    

    return accuracy

base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)

base_model.fit(train_features_transformed, train_labels)

test_features_transformed = fullpipeline.transform(test_features) 

base_accuracy = evaluate(base_model, test_features_transformed, test_labels)

               

               

best_random = rf_random.best_estimator_

random_accuracy = evaluate(best_random, test_features_transformed, test_labels)

               

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
# rmse_rf = np.sqrt(-scores_rf)

# display_scores(rmse_rf)
sampleSubmissiondf = pd.read_csv(datapath + "sample_submission.csv");

print(sampleSubmissiondf.head())

testdf = pd.read_csv(datapath + "test.csv");

testdf.head()

IDArr = testdf['Id'].values

clf = best_random

clf.fit(train_features_transformed, train_labels)

final_test_predictions = clf.predict(fullpipeline.transform(testdf))

df = pd.DataFrame({'Id':IDArr,'SalePrice': final_test_predictions})

# df.to_csv(index=False)



df.to_csv(r'results.csv',index=False)
df.describe()
testing_beta = fullpipeline.fit_transform(housing)

print(testing_beta[7])
print(df_ord_to_num.info())
print(cat_features)

print(num_features)
df2 = ColumnAdder().transform(housing[squareFeet_features])
print(df2)

# print(housing["Neighborhood"].unique())
# from sklearn.base import BaseEstimator

# from sklearn.base import TransformerMixin

# from sklearn.preprocessing import OneHotEncoder

# from sklearn.impute import SimpleImputer 



# class ReduceCategoriesEncoder(BaseEstimator, TransformerMixin):



#     def __init__(self, threshold=0):

#         self.threshold = threshold



#     def transform(self, X):

#         output = X.copy()

#         if self.threshold != 0:

#             counts = output.count()

#             for column in output.columns:

#                 counts= output[column].value_counts()

#                 repl = counts[counts <= self.threshold].index

#                 print(repl)

#                 output[column].replace(repl, 'uncommon',inplace=True)

#         return output



#     def fit(self, X, y=None):

#         return self