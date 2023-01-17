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
# Read csv file to our date frame named as 'df'.

df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv') 

# Check first 5 entried in our 'df' Data Frame.

df.head()
# Check bottom 5 entried in our 'df' Data Frame.

df.tail()
# Check information like no. of rows and columns, names of columns, non null entries 

# for each column and its dtype in our 'df' Data Frame.

df.info()
# Fetch features of type Object and print the result.

objFeatures = df.select_dtypes(include="object").columns

print (objFeatures)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



# Change dtype of columns.

for columnNames in objFeatures: # for each column it will iterate one at a time for 23 times (as 23 columns)

    df[columnNames] = le.fit_transform(df[columnNames].astype(str))
# dtypes got changed.

df.info()
# X and Y

X = df.drop(['class'], axis = 1) # Drop column 'Class' and select the rest columns in X.

y = df['class'] # Takes column 'Class' on y axis.              

X.info()
# Train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,

random_state=42)
X_train.info()
X_test.info()
# Model Training

from sklearn.naive_bayes import GaussianNB

# clf = # Empty Brain

clf = GaussianNB() # Declaring Rules

clf.fit(X_train, y_train) # Learning the dataset
#Predicting Y value

y_predicted_values = clf.predict(X_test)

print(y_predicted_values)