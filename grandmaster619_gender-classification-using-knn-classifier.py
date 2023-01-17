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
dataframe = pd.read_csv("/kaggle/input/gender-classification/Transformed Data Set - Sheet1.csv")

dataframe.info()
dataframe.rename(columns={'Favorite Color' :'FavoriteColor', 'Favorite Music Genre':'FavoriteMusicGenre', 

                          'Favorite Beverage':'FavoriteBeverage', 'Favorite Soft Drink':'FavoriteSoftDrink'}, inplace=True)
from sklearn.preprocessing import LabelEncoder

dataframe=dataframe.apply(LabelEncoder().fit_transform)

dataframe.info()

X = dataframe.drop(['Gender'], axis = 1)

y = dataframe.Gender
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)



y_pred = knn.predict(X_test)

print("Result for the test set is : {}".format(y_pred))

print("Score on training set is : {:.2f}%".format(knn.score(X_train, y_train)*100))

print("Score on test set is : {:.2f}%".format(knn.score(X_test, y_test)*100))
from sklearn.metrics import recall_score, precision_score, confusion_matrix
print("Recall score", recall_score(y_test, y_pred, average='macro'))

print("Precision score", precision_score(y_test, y_pred, average='macro'))

print ("CONFUSION MATRIX", confusion_matrix(y_test, y_pred))