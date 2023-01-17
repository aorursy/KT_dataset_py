%pip install jcopml
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer





from jcopml.pipeline import num_pipe, cat_pipe

from jcopml.utils import save_model, load_model

from jcopml.plot import plot_missing_value

from jcopml.feature_importance import mean_score_decrease





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/titanic/train.csv", index_col="PassengerId")

df.head()
plot_missing_value(df, return_df=True)
# drop unuseful data

df.drop(columns=["Cabin", "Ticket"], inplace=True)
plot_missing_value(df)

# there are still missing values, but still torelable
df.head()
df["isAlone"] = (df.SibSp == 0) & (df.Parch == 0)

df.head()
df.Age = pd.cut(df.Age, [0, 5, 12, 17, 25, 45, 65, 120], labels=["toddler", "children", "teenager", "adult_teenager", "adult", "senior", "super_senior"])

df.head()
df.Fare = pd.cut(df.Fare, [0, 25, 100, 600], labels=["economy", "business", "executive"])

df.head()
X = df.drop(columns="Survived")

y = df["Survived"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import RandomizedSearchCV

from jcopml.tuning import random_search_params as rsp

from jcopml.tuning import grid_search_params as gsp

from sklearn.model_selection import train_test_split, GridSearchCV
preprocessor = ColumnTransformer([

    ('numeric', num_pipe(scaling="minmax"), ["SibSp", "Parch"]),

    ('categoric', cat_pipe(encoder="onehot"), ["Pclass", "Sex", "Age", "Fare", "Embarked", "isAlone"]),

])





pipeline = Pipeline([

    ('prep', preprocessor),

#     ('algo', RandomForestClassifier(n_jobs=-1, random_state=42))

    ('algo', KNeighborsClassifier())

])



# model = RandomizedSearchCV(pipeline, rsp.rf_params, cv=5, n_iter=50, n_jobs=-1, verbose=1, random_state=42)

model = GridSearchCV(pipeline, gsp.knn_params, cv=3, n_jobs=-1, verbose=1)

model.fit(X_train, y_train)



print(model.best_params_)

print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
preprocessor = ColumnTransformer([

    ('numeric', num_pipe(scaling="minmax"), ["SibSp", "Parch"]),

    ('categoric', cat_pipe(encoder="onehot"), ["Pclass", "Sex", "Age", "Fare", "Embarked", "isAlone"]),

])





pipeline = Pipeline([

    ('prep', preprocessor),

    ('algo', RandomForestClassifier(n_jobs=-1, random_state=42))

])



# model = RandomizedSearchCV(pipeline, rsp.rf_params, cv=5, n_iter=50, n_jobs=-1, verbose=1, random_state=42)

model = GridSearchCV(pipeline, gsp.rf_params, cv=3, n_jobs=-1, verbose=1)

model.fit(X_train, y_train)



print(model.best_params_)

print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
from jcopml.utils import save_model
save_model(model, "titanic_v1.pkl")
def submit(model, filename="titanic-v1.csv"):

    df_submit = pd.read_csv("/kaggle/input/titanic/test.csv", index_col="PassengerId")

    df_submit.drop(columns=["Cabin", "Ticket"], inplace=True)

    df_submit["isAlone"] = (df_submit.SibSp == 0) & (df_submit.Parch == 0)

    df_submit.Age = pd.cut(df_submit.Age, [0, 5, 12, 17, 25, 45, 65, 120], labels=["toddler", "children", "teenager", "adult_teenager", "adult", "senior", "super_senior"])

    df_submit.Fare = pd.cut(df_submit.Fare, [0, 25, 100, 600], labels=["economy", "business", "executive"])

    df_submit['Survived'] = model.predict(df_submit)

    df_submit[['Survived']].to_csv(filename, index_label='PassengerId')
submit(model)