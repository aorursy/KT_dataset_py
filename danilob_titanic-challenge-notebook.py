# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #Plotting



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Loads data

train_filename = "../input/titanic/train.csv"

test_filename = "../input/titanic/test.csv"



train = pd.read_csv(train_filename, index_col=0)

test = pd.read_csv(test_filename, index_col=0)



train.head()
y = train.pop('Survived')

train.info()
#Feauture engineering

import re



def cluster_titles(title):

    if title in ['Mme.', 'Miss.', 'Ms.']:

        return 'Miss'

    elif title in ['Mr.', 'Mrs.', 'Master.']:

        return title

    else:

        return 'Rare'



def extract_cabin(cabin):

    if cabin != cabin:

        return 'NAN'

    else: 

        return re.findall('^(.)', cabin)[0]

    

def preprocess(df):

    #Compute names titles

    names = df['Name']

    titles = [re.findall(', (.+?) ', name)[0] for name in names]

    df['Titles'] = [cluster_titles(title) for title in titles]

    

    #Compute name lenght

    df['NameLen'] = df['Name'].apply(len)

    

    #Computes cabin letter

    df['CabinLetter'] = train.Cabin.map(extract_cabin)



    

preprocess(train)

preprocess(test)

train.head()
#Separate numeric and categorical columns

num_cols = ['Age', 'Fare', 'NameLen']

cat_cols = ['Pclass', 'SibSp', 'Parch', 'Sex', 'Embarked', 'Titles', 'CabinLetter']



cols = num_cols + cat_cols



print(f'Numeric cols: {num_cols}')

print(f'Categorical cols: {cat_cols}')
#Plot

fig = plt.figure(figsize=(20, 12))



for i, col in enumerate(cols):

    ax = plt.subplot(3, 4, i+1)

    if col in num_cols:

        train[cols].groupby(y)[col].hist(histtype='step',

                                        lw=2, density=True, ax=ax)

        plt.xlabel(col)

        plt.legend(['0', '1'])

        plt.yscale('log')

        

    if col in cat_cols:

        train[cols].groupby([y, col]).size().unstack(0).plot(kind='bar', ax=ax)

        plt.xlabel(col)

        plt.yscale('log')
#Model 

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer





#Make pipeline

cat_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



num_transformer = SimpleImputer(strategy='mean')



preprocessor = ColumnTransformer(

    transformers=[

        ('num', num_transformer, num_cols),

        ('cat', cat_transformer, cat_cols)

    ])



#Define model

rf = RandomForestClassifier(random_state=666)



pipe = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', rf)

                             ])
#Hyperparamter optimization

from sklearn.model_selection import GridSearchCV



param_grid = {

    'model__n_estimators': [100, 200, 500, 1000],

    'model__max_depth': [2, 5, 10, 50, 100],

}



search = GridSearchCV(pipe, param_grid, n_jobs=6, cv=10)

search.fit(train[cols], y)

print("Best parameter (CV score=%0.3f):" % search.best_score_)

print(search.best_params_)
pred = search.predict(test[cols])



sub = pd.DataFrame(list(zip(test.index, pred)))

sub.columns = ['PassengerId', 'Survived']

sub.to_csv('Submission.csv', index=False)