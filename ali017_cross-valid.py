import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_absolute_error

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor



train_data = pd.read_csv('../input/housing-prices-competition-for-kaggle-learn-users/train.csv')

test_data = pd.read_csv('../input/housing-prices-competition-for-kaggle-learn-users/test.csv')



y = train_data.SalePrice

train_data.dropna(subset=['SalePrice'], axis=0, inplace=True)

train_data.drop(['SalePrice'], axis=1, inplace=True)



numeric_cols = [col for col in train_data.columns if train_data[col].dtypes in ['int64', 'float64']]

categoric_cols = [col for col in train_data.columns if set(train_data[col])==set(test_data[col]) and train_data[col].nunique() < 10 and train_data[col].dtypes == 'object']

cols = numeric_cols + categoric_cols



print("Unique values in 'Condition2' column in training data:", train_data['MiscFeature'].unique())

print("Unique values in 'Condition2' column in training data:", test_data['MiscFeature'].unique())

print(categoric_cols)

X = train_data[cols].copy()

X_test = test_data[cols].copy()
numeric_trans = SimpleImputer(strategy='mean')

categoric_trans = Pipeline(steps = [

                                    ('impute', SimpleImputer(strategy='most_frequent')),

                                    ('One-Hot', OneHotEncoder(handle_unknown='ignore'))

])

preprocessor = ColumnTransformer(transformers = [

                                                ('numeric', numeric_trans, numeric_cols),

                                                ('categoric', categoric_trans, categoric_cols)

])

def score_n(n_estimators):

    my_pipeline = Pipeline(steps = [

                                   ('preprocess', preprocessor),

                                   ('model', RandomForestRegressor(n_estimators, random_state=0))

    ])

    #scores = []

    scores = -1 * cross_val_score(my_pipeline, X, y, cv = 5, scoring='neg_mean_absolute_error')

    return scores.mean()
results = {}

#print(type(results))

for i in range(1, 6):

    results[50*i] = score_n(50*i)

    #print(type(score_n))

import matplotlib.pyplot as plt

%matplotlib inline



plt.plot(results.keys(), results.values())

plt.show()
a = min(results, key=results.get)

print(a)