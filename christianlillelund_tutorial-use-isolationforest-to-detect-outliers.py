import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import warnings



warnings.filterwarnings('ignore')

pd.reset_option('^display.', silent=True)



# Load the two datasets

X_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

X_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



# Seperate independent and dependent variable

num_train = len(X_train)

num_test = len(X_test)

y_train = X_train.SalePrice

X_train.drop(['SalePrice'], axis=1, inplace=True)



# Merge train and test data to simplify processing

df = pd.concat([X_train, X_test], ignore_index=True)



# Rename odd-named columns

df = df.rename(columns={"1stFlrSF": "FirstFlrSF",

                        "2ndFlrSF": "SecondFlrSF",

                       "3SsnPorch": "ThirdSsnPorch"})



# Shopw 5 samples

df.head()
# Find columns with more than 1000 NaN's and drop them (see above)

columns = [col for col in df.columns if df[col].isnull().sum() > 1000]

df = df.drop(columns, axis=1)



# Fill LotFrontage with median

df['LotFrontage'].fillna((df['LotFrontage'].mean()), inplace=True)



# No garage values means no year, area or cars

for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:

    df[col] = df[col].fillna(0)

    

# No garage info means you don't have one

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    df[col] = df[col].fillna('None')



# Fill no basement

for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:

    df[col] = df[col].fillna('None')



# Fill remaining categorical and numerical cols with None and 0

cat_columns = df.select_dtypes('object').columns

num_columns = [i for i in list(df.columns) if i not in cat_columns]

df.update(df[cat_columns].fillna('None'))

df.update(df[num_columns].fillna(0))



# Check for missing values 

print(df.isnull().values.any())
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(0)



# Helper function to train and predict IF model for a feature

def train_and_predict_if(df, feature):

    clf = IsolationForest(max_samples=100, random_state=rng)

    clf.fit(df[[feature]])

    pred = clf.predict(df[[feature]])

    scores = clf.decision_function(df[[feature]])

    stats = pd.DataFrame()

    stats['val'] = df[feature]

    stats['score'] = scores

    stats['outlier'] = pred 

    stats['min'] = df[feature].min()

    stats['max'] = df[feature].max()

    stats['mean'] = df[feature].mean()

    stats['feature'] = [feature] * len(df)

    return stats



# Helper function to print outliers

def print_outliers(df, feature, n):

    print(feature)

    print(df[feature].head(n).to_string(), "\n")



# Run through all features and save the outlier scores for each feature

num_columns = [i for i in list(df.columns) if i not in list(df.select_dtypes('object').columns) and i not in ['Id']]

result = pd.DataFrame()

for feature in num_columns:

    stats = train_and_predict_if(df, feature)

    result = pd.concat([result, stats])

    

# Gather top outliers for each feature

outliers = {team:grp.drop('feature', axis=1) 

       for team, grp in result.sort_values(by='score').groupby('feature')}



# Print the top 10 outlier samples for a few selected features

n_outliers = 10

print_outliers(outliers, "LotArea", n_outliers)

print_outliers(outliers, "YearBuilt", n_outliers)

print_outliers(outliers, "BsmtUnfSF", n_outliers)

print_outliers(outliers, "GarageYrBlt", n_outliers)
# Use clipping to level out the observed outliers in the data

df.LotArea = df.LotArea.clip(1300,50000)

df.YearBuilt = df.YearBuilt.clip(1880,2010)

df.BsmtUnfSF = df.BsmtUnfSF.clip(100,1900)

df.GarageYrBlt = df.GarageYrBlt.clip(0,2020)
# Train IsolationForest again, this time after with clipped outliers for 4 features

result = pd.DataFrame()

for feature in num_columns:

    stats = train_and_predict_if(df, feature)

    result = pd.concat([result, stats])



# Gather top outliers for each feature

outliers = {team:grp.drop('feature', axis=1) 

       for team, grp in result.sort_values(by='score').groupby('feature')}





# Print the top 10 outlier samples for the features we chose to clip

n_outliers = 10

print_outliers(outliers, "LotArea", n_outliers)

print_outliers(outliers, "YearBuilt", n_outliers)

print_outliers(outliers, "BsmtUnfSF", n_outliers)

print_outliers(outliers, "GarageYrBlt", n_outliers)