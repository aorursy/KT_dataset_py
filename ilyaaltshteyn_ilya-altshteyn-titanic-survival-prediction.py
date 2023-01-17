import numpy as np
import pandas as pd

df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.tail(20)

from matplotlib import pyplot as plt

plt.hist(df['Age'])
plt.vlines(df['Age'].median(), ymin=0, ymax=200)
plt.vlines(df['Age'].mean(), ymin=0, ymax=200, color='red')

# build features!
from sklearn.preprocessing import OneHotEncoder
import re

def get_test_data():
    return pd.read_csv('/kaggle/input/titanic/test.csv')

def get_train_data():
    return pd.read_csv('/kaggle/input/titanic/train.csv')

def _is_a_miss(datapoint):
    return 'miss' in str(datapoint).lower()

def _is_a_mrs(datapoint):
    return 'mrs' in str(datapoint).lower()
    
def _age_is_missing(row):
    try: 
        return np.isnan(row.Age)
    except:
        return False
    
def _cabin_missing(cabin):
    try:
        if np.isnan(cabin):
            return True
        return False
    except:
        return False

def _letters_in_ticket_number(ticket):
    return not ticket.isdigit()
    
    
def _make_categorical_features(df, test):
    """ Takes in a raw dataframe and returns categorical feature columns only. 
    """ 
    train_dont_touch = get_train_data()
    test_dont_touch = get_test_data()
    
    for temp_df in [df, train_dont_touch, test_dont_touch]:
        temp_df['is_a_miss'] = temp_df['Name'].apply(_is_a_miss)
        temp_df['is_a_mrs'] = temp_df['Name'].apply(_is_a_mrs)
        temp_df['age_is_missing'] = temp_df.apply(_age_is_missing, axis=1)
        temp_df['cabin_missing'] = temp_df['Cabin'].apply(_cabin_missing)
        temp_df['letters_in_ticket_number'] = temp_df.Ticket.apply(_letters_in_ticket_number)
    
    categoricals = [
        'Pclass',  # native
        'Sex',  # native
        'Embarked',  # native
        'is_a_miss',  # from here down, engineered features
        'is_a_mrs', 
        'age_is_missing',
        'cabin_missing',
        'letters_in_ticket_number'
    ]

    if not test:
        # create encoder using combined train + test data
        test_data_categoricals = train_dont_touch[categoricals]
        train_data_categoricals = test_dont_touch[categoricals]
        combined_categorical_data = pd.concat(
            [test_data_categoricals, train_data_categoricals]
        )

        columns_to_encode = combined_categorical_data.fillna('nan')
        
        global enc
        enc = OneHotEncoder(drop='first', sparse=False).fit(columns_to_encode)
        
    cats = pd.DataFrame(enc.transform(df[categoricals].fillna(value='nan')))
    
    return cats

def _make_numerical_features(df):
    df['Age'] = df['Age'].fillna(value=df['Age'].mean())
    
    numeric_colnames = [
        'SibSp',  # native
        'Parch',  # native
        'Fare',  # native
        'Age',  # native
    ]
    numerics = df[numeric_colnames]
    
    return numerics
    

def get_prepped_data(df, test=False):
    cats = _make_categorical_features(df, test=test)
    noncats = _make_numerical_features(df)

    X = pd.concat([cats, noncats], axis=1)
    if test:
        return X
    
    # training data!
    y = df['Survived']
    return X, y

X, y = get_prepped_data(df, test=False)
X
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

params = {
    'n_estimators': [10, 25, 75],
    'max_depth': [None, 10, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3]
}

clf = GridSearchCV(rf, params, verbose=10, n_jobs=-1)

clf.fit(X, y)
print(f'best score: {clf.best_score_}')
print(f'best score: {clf.best_params_}')
from datetime import datetime as dt

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
X = get_prepped_data(df_test, test=True)

df_test['Survived'] = clf.predict(X.fillna(method='ffill'))
df_test[['PassengerId', 'Survived']].to_csv(f'submission {dt.now()}.csv', index=False)

enc.categories_
