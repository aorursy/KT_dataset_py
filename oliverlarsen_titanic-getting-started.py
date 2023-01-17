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
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")



# Personal preference, eases future use of referring to train_data & test_data.

df_train = train_data

df_test = test_data



# Defining y, only the training set provides data for survival, Kaggle's test set doesn't.

y_train = df_train["Survived"].copy()



# Defining x, for both the training and test sets.

X_train = df_train[["Sex", "Pclass", "Embarked", "SibSp", "Parch"]].copy()

X_test = df_test[["Sex", "Pclass", "Embarked", "SibSp", "Parch"]].copy()



# Transforming the Sex & Embarked variables to catagorical for both sets.

X_train['Sex_cat'] = X_train['Sex'].astype('category')

X_test['Sex_cat'] = X_test['Sex'].astype('category')



X_train['Embarked_cat'] = X_train['Embarked'].astype('category')

X_test['Embarked_cat'] = X_test['Embarked'].astype('category')



# Changing the Sex & Embarked categorical to codes for both sets, storing them in a new variable as to not disturb the original.

X_train['Sex_cat_codes'] = X_train['Sex_cat'].cat.codes

X_test['Sex_cat_codes'] = X_test['Sex_cat'].cat.codes



X_train['Embarked_cat_codes'] = X_train['Embarked_cat'].cat.codes

X_test['Embarked_cat_codes'] = X_test['Embarked_cat'].cat.codes



# Deleting the unnecessary columns, as this model only requires the final column Sex_cat_codes & Embarked_cat_codes.

del X_train['Sex']

del X_test['Sex']



del X_train['Sex_cat']

del X_test['Sex_cat']



del X_train['Embarked']

del X_test['Embarked']



del X_train['Embarked_cat']

del X_test['Embarked_cat']



# Instantiate and fitting the classifier, which we'll call model.

model = LogisticRegression(max_iter=500)

model.fit(X=X_train, y=y_train.values.ravel())



# Using the model to predict.

predictions = model.predict(X_test)



# Writing to a file, which allows for submission on the Kaggle platform.

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")