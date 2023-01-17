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
df = pd.read_csv("../input/gender-classification/Transformed Data Set - Sheet1.csv")
df.head()
df.info()
df.Gender[df.Gender == "F"].count()
df.Gender = [1 if each=="F" else 0 for each in df.Gender]
#Creating new columns with get_dummines to work with Logistic Regression

columns_color = pd.get_dummies(df["Favorite Color"], prefix='color')

columns_music = pd.get_dummies(df["Favorite Music Genre"], prefix='music')

columns_beverage = pd.get_dummies(df["Favorite Beverage"], prefix='beverage')

columns_drink = pd.get_dummies(df["Favorite Soft Drink"], prefix='drink')



final = pd.concat([df, columns_color], axis=1)

final = pd.concat([final, columns_music], axis=1)

final = pd.concat([final, columns_beverage], axis=1)

final = pd.concat([final, columns_drink], axis=1)

final.drop(["Favorite Color", "Favorite Music Genre", "Favorite Beverage", "Favorite Soft Drink"], axis=1, inplace=True)
x = final.drop("Gender", axis=1)

y = final["Gender"]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train, y_train)
lr.score(x_test, y_test)