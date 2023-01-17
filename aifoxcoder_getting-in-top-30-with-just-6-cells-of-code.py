import pandas as pd

import numpy as np

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
data_dir = '/kaggle/input/titanic/'



train = pd.read_csv(data_dir + 'train.csv')

test = pd.read_csv(data_dir + 'test.csv')

test_id = test.PassengerId

sample_submission = pd.read_csv(data_dir + 'gender_submission.csv')



train.head()
train.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)  # All of them are kind of unique identifiers, hence useless for our model

test.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)



def train_cats(df):     #Turns Object datatype to Category datatype

    for col in df.columns:

        if df[col].dtype == 'O' : 

            df[col] = df[col].astype('category').cat.as_ordered()

            

train_cats(train)

train_cats(test)

train.info()
def fix_missing(df):  #Fixes Missing Values

    for col in df.columns:

        if (df[col].dtype == 'int64') or (df[col].dtype == 'float64') or (df[col].dtype == 'bool'):

            if df[col].isnull().sum() != 0:

                df[col + '_na'] = df[col].isnull()

                df[col] = df[col].fillna(df[col].median())

        else:

            df[col + '_coded'] = df[col].cat.codes +1

            df.drop(columns=[col], axis=1, inplace=True)

            

fix_missing(train)

fix_missing(test)

train.info()
X = train.drop(columns='Survived')

y = train.Survived



rfc = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=10) #default settings

cross_val_score(rfc, X, y, cv=7, n_jobs=-1, verbose=1).mean()
rfc.fit(X, y)

sub = rfc.predict(test.drop(columns='Fare_na'))  #Because 

sub_df = pd.DataFrame(data={'PassengerId': test_id, 'Survived': sub})

sub_df.to_csv('submission.csv', index=False)