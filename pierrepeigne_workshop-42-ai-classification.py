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
data = pd.read_csv('/kaggle/input/titanic/train.csv')
data = data.set_index("PassengerId")
data
data.Survived.mean()
data["Female"] = (data["Sex"] == "female")
data["Female"] *= 1 
data = data.drop("Sex", axis=1)
data = data.drop(["Name", "Cabin", "Ticket"], axis=1)
import matplotlib.pyplot as plt

import seaborn as sns
sns.violinplot(x="Female", y="Age", hue="Survived", data=data)
sns.boxplot(x="Survived", y="Fare", hue="Pclass", data=data)
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
## TODO: Handle outliers in the numerical features before normalization 
numbers = data.select_dtypes(exclude="object").drop("Pclass", axis=1).columns

categories = data.select_dtypes("object").columns



categories = list(categories) + ["Pclass"]



numbers = numbers.drop("Survived")
# Imputer 

imputer = SimpleImputer(strategy="median")



# Scaler

scaler = StandardScaler()



steps = [('imputer', imputer), ("scaler", scaler)]

pipe = Pipeline(steps)
y = data["Survived"]

x_num = data.drop("Survived", axis=1)
x_num = pipe.fit_transform(x_num[numbers])
x_num = pd.DataFrame(x_num, columns=numbers)
data[categories]
x_cat = data[categories]

x_cat["Pclass"] = x_cat["Pclass"].astype(object)



x_cat = pd.get_dummies(x_cat)



preprocessed = pd.concat(([x_num.set_index(data.index), x_cat]), axis=1)
preprocessed
## TODO Make a real pipeline using sklearn.pipeline for the categorical features
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(preprocessed, y)



model = LogisticRegression().fit(X_train, y_train)



pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy_score(pred, y_test)
precision_score(pred, y_test)
recall_score(pred, y_test)
f1_score(pred, y_test)
data.corr()
data2 = pd.read_csv('/kaggle/input/titanic/train.csv')
data2.Cabin.unique()
## TODO: extract the letter from the cabin values and use it as a categorical feature. (Do not forget to handle Nan values! :)) 

## Handling Nan: 

## - try with the median 

## - try as a new letter (e.g: 'Z')



# Retrain the algo with this new feature