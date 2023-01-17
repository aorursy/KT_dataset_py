# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



def load_data(path):

    return pd.read_csv(path)



def transform():

#     print(ds_train.shape)

    ds_train.drop(columns = ['Name', 'Ticket'], inplace=True)

    







# ROOT CODE

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        if('train' in filename):

            ds_train = load_data(os.path.join(dirname, filename))

        elif('test' in filename):

            ds_test = load_data(os.path.join(dirname, filename))

            ds_test_bu = load_data(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print('Train : ',ds_train.shape)

print('Test : ',ds_test.shape)
ds_train.info()
# Remove unwanted columns

ds_train.drop(columns = ['PassengerId'], inplace=True)

ds_test.drop(columns = ['PassengerId'], inplace=True)
# Check shape after column drop

ds_train.shape
ds_test.shape
ds_train['Sex'].value_counts()
# Converting sex to numeric values

# ds_train['Male'] = np.where(ds_train['Sex']=='male', 1, 0)
# ds_train['NamLen']  = [None if x ==None else len(x) for x in ds_train['Name']] 
# ds_train['Cabin'].value_counts()
# ds_train['Deck'] = [None if x==None else str(str(x)[0]).upper() for x in ds_train['Cabin']]

# ds_train['Deck'].value_counts()
def prep_data(df):

    df['Male'] = np.where(df['Sex']=='male', 1, 0)

    df['NamLen']  = [None if x ==None else len(x) for x in df['Name']] 

    df['Deck'] = [None if x==None else str(str(x)[0]).upper() for x in df['Cabin']]

    return df
ds_train = prep_data(ds_train)
ds_train.shape
# Co-relation Matrix

corr = ds_train.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
ds_train.hist(bins=50, figsize=(20,15))
ds_train.round({'Age':0, 'Fare':0, })
ds_train.info()
y = ds_train[['Survived']]

X = ds_train.drop('Survived', axis=1)



print('X shape :', X.shape)

print('y shape :', y.shape)
from sklearn.base import BaseEstimator, TransformerMixin



class TitanicDataCatgorization(BaseEstimator, TransformerMixin):

    def __init__(self):

        return

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        deck_dummies = pd.get_dummies(X['Deck'] ,prefix='deck', drop_first=True)

        embarked_dummies = pd.get_dummies(X['Embarked'] ,prefix='embarked', drop_first=True)

#         clarity_dummies = pd.get_dummies(X['clarity'] ,prefix='clarity', drop_first=True)

        

        X = pd.concat([X, deck_dummies, embarked_dummies], axis=1)

        X.drop(['Deck', 'Embarked'], axis=1, inplace=True)

        

        return X
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder



cat_pipeline = Pipeline([

    ('catrizer', TitanicDataCatgorization())

])



cat_pipeline_2 = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])



num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy="median")),

    ('std_scaler', StandardScaler())

])



from sklearn.compose import ColumnTransformer



num_attr = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Male", "NamLen"]

cat_attr = ["Embarked", "Deck"]



full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attr),

        ("cat", cat_pipeline_2, cat_attr),

    ])
X_p = full_pipeline.fit_transform(X)
X_p.shape
def display_scores(score):

    print("Scores:",score)

    print("Mean:", score.mean())

    print("Standard dviation:", score.std())



from sklearn.model_selection import cross_val_score



# RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier



forest_reg = RandomForestClassifier()

forest_reg.fit(X_p, y)



forest_reg_score = cross_val_score(forest_reg, X_p, y, scoring="neg_mean_squared_error", cv=4)

forest_reg_rmse_score = np.sqrt(-forest_reg_score)



display_scores(forest_reg_rmse_score)
from sklearn.metrics import mean_squared_error



predictions_training = forest_reg.predict(X_p)



train_mse = mean_squared_error(y, predictions_training)

train_rmse = np.sqrt(train_mse)



print("Training RMSE :", train_rmse)
ds_test.hist(bins=50, figsize=(20,15))
ds_test.shape
# TEST



X_test = prep_data(ds_test)

X_test.info()
X_test.shape
X_test_p = full_pipeline.transform(X_test)

X_test_p.shape
predictions_test = forest_reg.predict(X_test_p)
predictions_test
predict_ds = pd.DataFrame({'PassengerId': ds_test_bu['PassengerId'], 'Survived': predictions_test})
predict_ds.info()
file_name='submission.csv'

predict_ds.to_csv(file_name, encoding='utf-8', index=False)