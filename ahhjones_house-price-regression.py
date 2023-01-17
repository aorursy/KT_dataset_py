import os

from pathlib import Path



import numpy as np

import pandas as pd

from scipy.stats import norm



from sklearn.base import TransformerMixin

from sklearn.preprocessing import OrdinalEncoder

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, make_scorer



import matplotlib.pyplot as plt

import seaborn as sns



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_path = Path('/kaggle/input/house-prices-advanced-regression-techniques')

train_df = pd.read_csv(data_path / 'train.csv', index_col=['Id'])

test_df = pd.read_csv(data_path / 'test.csv', index_col=['Id'])
train_df.head()
test_df.head()
# Plot sale price histogram

sns.distplot(train_df.SalePrice, fit=norm);
sns.distplot(np.log(train_df.SalePrice), fit=norm)
area_cols = [col for col in train_df if 'area' in col.lower()]

fig, ax = plt.subplots(1, len(area_cols), figsize=(20, 5)) 

for i, col in enumerate(area_cols):

    sns.distplot(train_df[col].dropna(), ax=ax[i], fit=norm)
fig, ax = plt.subplots(1, len(area_cols), figsize=(20, 5)) 

for i, col in enumerate(area_cols):

    sns.distplot(np.log(train_df.loc[train_df[col] > 0, col]), ax=ax[i], fit=norm)
sf_cols = [col for col in train_df if 'sf' in col.lower()]

fig, ax = plt.subplots(1, len(sf_cols), figsize=(20, 3)) 

for i, col in enumerate(sf_cols):

    sns.distplot(train_df[col].dropna(), ax=ax[i], fit=norm)
fig, ax = plt.subplots(1, len(sf_cols), figsize=(20, 3)) 

for i, col in enumerate(sf_cols):

    sns.distplot(np.log(train_df.loc[train_df[col] > 0, col]), ax=ax[i], fit=norm)
missing_fractions = train_df.isnull().sum() / len(train_df.index)



# Plot top 10 missing value columns.

top_missing = missing_fractions.sort_values().tail(10)

sns.scatterplot(top_missing.values, top_missing.index)
# Log transform of SalePrice

train_df['LogSalePrice'] = np.log(train_df['SalePrice'])
# Drop columns with >80% of values missing. (include MiscVal since it is accociated with MiscFeature)

drop_cols = ['PoolQC', 'MiscFeature', 'MiscVal', 'Alley', 'Fence']

train_df.drop(drop_cols, axis=1, inplace=True)

test_df.drop(drop_cols, axis=1, inplace=True)
# Thanks very much to: https://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn

class DataFrameImputer(TransformerMixin):



    def __init__(self):

        """Impute missing values.



        Columns of dtype object are imputed with the most frequent value 

        in column.



        Columns of other types are imputed with mean of column.



        """

    def fit(self, X, y=None):



        self.fill = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],

            index=X.columns)



        return self



    def transform(self, X, y=None):

        return X.fillna(self.fill)

    

imputer = DataFrameImputer()

train_df = imputer.fit_transform(train_df)

test_df = imputer.transform(test_df)
def preprocess_area_columns(df):

    df = df.copy()

    df['HasPool'] = (df['PoolArea'] > 0).astype(int)

    df.drop(['PoolArea'], axis=1, inplace=True)  # Drop original column

    df['HasMasVnr'] = (df['MasVnrArea'] > 0).astype(int)

    df.drop(['MasVnrArea'], axis=1, inplace=True)  # Drop original column

    log_cols = ['LotArea', 'GrLivArea', 'GarageArea']

    df[log_cols] = np.where(df[log_cols] > 0, np.log(df[log_cols]), 0)

    return df



train_df = preprocess_area_columns(train_df)

test_df = preprocess_area_columns(test_df)
def preprocess_sf_columns(df):

    df = df.copy()

    df['HasBmst'] = (df['TotalBsmtSF'] > 0).astype(int)

    df['Has2ndFlr'] = (df['2ndFlrSF'] > 0).astype(int)

    df['HasDeck'] = (df['WoodDeckSF'] > 0).astype(int)

    df['HasPorch'] = (df['OpenPorchSF'] > 0).astype(int)

    log_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'WoodDeckSF', 'OpenPorchSF']

    df[log_cols] = np.where(df[log_cols] > 0, np.log(df[log_cols]), 0)

    return df



train_df = preprocess_sf_columns(train_df)

test_df = preprocess_sf_columns(test_df)
# Columns to be encoded

encode_cols = [col for col in train_df if train_df[col].dtype == np.dtype('O')]



# Train encoder on concatenated training and test sets.

encoder = OrdinalEncoder()

encoder.fit(pd.concat([train_df, test_df], sort=False)[encode_cols])



# Encode training and test sets

train_df[encode_cols] = encoder.transform(train_df[encode_cols])

test_df[encode_cols] = encoder.transform(test_df[encode_cols])

train_df[encode_cols].head()
feature_cols = [col for col in train_df if col not in ['SalePrice', 'LogSalePrice']]



X = train_df[feature_cols].values

y = train_df.LogSalePrice.values



X.shape
# Used for leaderboard evaluation.

def root_mean_squared_error(*args):

    return mean_squared_error(*args) ** 0.5



# Scoring function for cross-validation.

scorer = make_scorer(root_mean_squared_error, greater_is_better=False)



# Grid search.

model = GradientBoostingRegressor(loss='ls', criterion='mse', random_state=0, subsample=0.9, max_depth=3)

param_grid = {

    'n_estimators': [550, 600, 650],

    'min_samples_split': [3, 4],

}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=5, refit=True, n_jobs=-1)

grid_search.fit(X, y)



print(f'Best RMSE: {-grid_search.best_score_:.3f}')

print(f'Best estimator: {grid_search.best_estimator_}')
feature_importances = grid_search.best_estimator_.feature_importances_

order = np.argsort(feature_importances)[::-1]  # Sort descending

for col, importance in zip(train_df.columns[order], feature_importances[order]):

    print(col, f'{importance:.5f}')
# Get best refitted estimator.

model = grid_search.best_estimator_



# Generate predictions.

X_test = test_df[feature_cols].values

log_y_pred = model.predict(X_test)



# Convert back to linear scale

y_pred = np.exp(log_y_pred)
sample_sub_df = pd.read_csv(data_path / 'sample_submission.csv')



sub_df = pd.DataFrame({

    'Id': sample_sub_df['Id'].values.tolist(),

    'SalePrice': y_pred.ravel()

})



sub_df.to_csv('submission.csv', index=False)

sub_df.head()