# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/mushrooms.csv')

df.head()
# Check which columns need to be hot encoded vs label encoded

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

labelencoder = LabelEncoder()

onehotencoder = OneHotEncoder()



redundant_columns = []

for column in df.columns:

    if df[column].nunique() > 2:

        column_names = [f'{column}_{category}' for category in df[column].unique()]

        df = df.join(

            pd.DataFrame( onehotencoder.fit_transform( df[[column]]).toarray(), columns=column_names )

        )

        redundant_columns.append(column)

    else:

        df[column] = labelencoder.fit_transform(df[column])



for column in redundant_columns:

    df = df.drop(column, axis=1)

df.head()
# print( *redundant_columns )



# for column in df.columns:

#     print(column, df[column].unique())

# df[df.columns[1:]]
from sklearn.model_selection import train_test_split

X = df[df.columns[1:]]

Y = df['class']



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
from sklearn.linear_model import LinearRegression

from sklearn import model_selection



model = LinearRegression()



kfold = model_selection.KFold(n_splits=10)



scoring = 'neg_mean_absolute_error'

results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print("MAE:", results.mean(), results.std() )



scoring = 'neg_mean_squared_error'

results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print("MSE:", results.mean(), results.std() )



scoring = 'r2'

results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print("R2 :", results.mean(), results.std() )
from sklearn.preprocessing import PolynomialFeatures



model = LinearRegression()

poly = PolynomialFeatures(degree=2)



x_poly = poly.fit_transform(X)

y_poly = poly.fit_transform([Y])





kfold = model_selection.KFold(n_splits=10)



scoring = 'neg_mean_absolute_error'

results = model_selection.cross_val_score(model, x_poly, Y, cv=kfold, scoring=scoring)

print("MAE:", results.mean(), results.std() )



scoring = 'neg_mean_squared_error'

results = model_selection.cross_val_score(model, x_poly, Y, cv=kfold, scoring=scoring)

print("MSE:", results.mean(), results.std() )



scoring = 'r2'

results = model_selection.cross_val_score(model, x_poly, Y, cv=kfold, scoring=scoring)

print("R2 :", results.mean(), results.std() )
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline



model = make_pipeline(StandardScaler(), SVC(gamma='auto'))



kfold = model_selection.KFold(n_splits=10)



scoring = 'neg_mean_absolute_error'

results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print("MAE:", results.mean(), results.std() )



scoring = 'neg_mean_squared_error'

results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print("MSE:", results.mean(), results.std() )



scoring = 'r2'

results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print(" R2:", results.mean(), results.std() )
from sklearn.cluster import KMeans



model = KMeans(n_clusters=2, random_state=0)



kfold = model_selection.KFold(n_splits=10)



scoring = 'neg_mean_absolute_error'

results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print("MAE:", results.mean(), results.std() )



scoring = 'neg_mean_squared_error'

results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print("MSE:", results.mean(), results.std() )



scoring = 'r2'

results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print(" R2:", results.mean(), results.std() )
from sklearn.naive_bayes import GaussianNB



model = GaussianNB()



kfold = model_selection.KFold(n_splits=10)



scoring = 'neg_mean_absolute_error'

results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print("MAE:", results.mean(), results.std() )



scoring = 'neg_mean_squared_error'

results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print("MSE:", results.mean(), results.std() )



scoring = 'r2'

results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print(" R2:", results.mean(), results.std() )