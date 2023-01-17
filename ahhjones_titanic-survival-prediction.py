import os

import re

from pathlib import Path



import numpy as np

import pandas as pd 

from scipy.stats import norm



import matplotlib.pyplot as plt

import seaborn as sns



from collections import Counter



from sklearn.base import TransformerMixin

from sklearn.preprocessing import OrdinalEncoder

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, make_scorer



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_path = Path('/kaggle/input/titanic')

train_df = pd.read_csv(data_path / 'train.csv', index_col='PassengerId')

test_df = pd.read_csv(data_path / 'test.csv', index_col='PassengerId')
train_df.head()
test_df.head()
train_df.Survived.value_counts() / train_df.index.size * 100
pd.DataFrame({

    'train': train_df.isnull().mean(), 

    'test': test_df.isnull().mean()

}) * 100
sns.distplot(train_df['Age'].dropna(), fit=norm)
sns.distplot(train_df['Fare'].dropna(), fit=norm)
sns.distplot(np.log(train_df.loc[train_df.Fare > 0].Fare), fit=norm)
# Assume title contains full stop. If multiple full stops, assume first value is title.

train_df['Title'] = train_df.Name.apply(lambda x: [y for y in re.split('[,\s]', x) if '.' in y][0])

test_df['Title'] = test_df.Name.apply(lambda x: [y for y in re.split('[,\s]', x) if '.' in y][0])



# Replace least common titles with 'Other'

most_common_titles = dict(Counter(train_df.Title.values).most_common(4))

train_df['Title'] = train_df.Title.apply(lambda x: x if x in most_common_titles else 'Other')

test_df['Title'] = test_df.Title.apply(lambda x: x if x in most_common_titles else 'Other')



train_df.head()
def get_cabin_number(x):

    if str(x) == 'nan':

        # No cabin

        return -1

    else:

        # First cabin number (i.e. if multiple cabins)

        try:

            return int(re.findall('[0-9]+', x)[0])

        # No Number found

        except IndexError:

            return None

    

    

def get_cabin_letter(x):

    if str(x) == 'nan':

        return 'No%20Cabin'

    else:

        return str(x)[0]

    

    

def prepare_cabin(df):

    df['HasCabin'] = (~df.Cabin.isnull()).astype(int)

    df['CabinNumber'] = df.Cabin.apply(get_cabin_number)

    df['CabinLetter'] = df.Cabin.apply(get_cabin_letter)

    return df





train_df = prepare_cabin(train_df)

test_df = prepare_cabin(test_df)

train_df.head()
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

train_df.head()
# Fit encoder.

encode_cols = ['Sex', 'Embarked', 'Title', 'CabinLetter']

encoder = OrdinalEncoder()

encoder.fit(pd.concat([train_df, test_df], sort=False)[encode_cols])



# Encode

train_df[encode_cols] = encoder.transform(train_df[encode_cols])

test_df[encode_cols] = encoder.transform(test_df[encode_cols])
# Age

train_df['Age'] = pd.qcut(train_df['Age'], 10, duplicates='drop', labels=False)

test_df['Age'] = pd.qcut(test_df['Age'], 10, duplicates='drop', labels=False)



# Fare

train_df['Fare'] = pd.qcut(train_df['Fare'], 10, duplicates='drop', labels=False)

test_df['Fare'] = pd.qcut(test_df['Fare'], 10, duplicates='drop', labels=False)
feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'HasCabin', 'CabinNumber', 'CabinLetter']

X = train_df[feature_cols].values

y = train_df.Survived
# Scoring function for cross-validation.

scorer = make_scorer(accuracy_score, greater_is_better=True)



# Grid search.

model = GradientBoostingClassifier(loss='deviance', criterion='friedman_mse', random_state=0, subsample=0.9)

param_grid = {

    'n_estimators': [100, 200, 300],

    'min_samples_split': [2, 3, 4],

    'max_depth': [3, 4, 5],

}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=5, refit=True, n_jobs=-1)

grid_search.fit(X, y)



print(f'Best accuracy: {grid_search.best_score_:.3f}')

print(f'Best estimator: {grid_search.best_estimator_}')
model = grid_search.best_estimator_

X_test = test_df[feature_cols].values

y_pred = model.predict(X_test)
samble_sub_df = pd.read_csv(data_path / 'gender_submission.csv')

samble_sub_df.head()
sub_df = pd.DataFrame({

    'PassengerId': test_df.index.values,

    'Survived': y_pred

})

sub_df.head()
sub_df.to_csv('submission.csv', index=False)