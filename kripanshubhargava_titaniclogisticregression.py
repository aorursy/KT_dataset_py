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
import pandas as pd

import numpy as np
PATH_DATA = '../input/titanic/'

titanic_train_df = pd.DataFrame(pd.read_csv("/kaggle/input/titanic/train.csv"))

titanic_test_df = pd.DataFrame(pd.read_csv("/kaggle/input/titanic/test.csv"))

gender_sub_df = pd.DataFrame(pd.read_csv("/kaggle/input/titanic/gender_submission.csv"))

pd.options.display.max_columns = None

titanic_train_df
titanic_train_df.dtypes
# Data cleaning

titanic_train_df['Embarked'] = titanic_train_df['Embarked'].replace(np.nan, "", regex=True)

titanic_train_df['Cabin'] = titanic_train_df['Cabin'].replace(np.nan, "", regex=True)

titanic_train_df['Age'] = titanic_train_df['Age'].replace(np.nan,titanic_train_df['Age'].median() , regex=True) #might drop rows if age is null



# test file

titanic_test_df['Embarked'] = titanic_test_df['Embarked'].replace(np.nan, "", regex=True)

titanic_test_df['Cabin'] = titanic_test_df['Cabin'].replace(np.nan, "", regex=True)

titanic_test_df['Age'] = titanic_test_df['Age'].replace(np.nan,titanic_test_df['Age'].median() , regex=True) #might drop rows if age is null
#label encoding for `Embarked`

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

titanic_train_df['Embarked'] = labelencoder.fit_transform(titanic_train_df['Embarked'])

titanic_train_df['Sex'] = labelencoder.fit_transform(titanic_train_df['Sex'])



# test file

titanic_test_df['Embarked'] = labelencoder.fit_transform(titanic_test_df['Embarked'])

titanic_test_df['Sex'] = labelencoder.fit_transform(titanic_test_df['Sex'])
titanic_test_df
# drop unnecessary columns and make x_train (X) and y_train

new_df = titanic_train_df.drop(columns=["Name","Ticket", "Fare", "PassengerId","Cabin", "Survived"])

new_test_df = titanic_test_df.drop(columns=["Name","Ticket", "Fare", "PassengerId","Cabin"])

new_df

new_test_df
# apply logistic regression Y = Survived , X = [PassangerId, ...]

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



X = new_df

y = titanic_train_df['Survived']

logisticRegr = LogisticRegression()

logisticRegr.fit(X, y)







# test dataset

y_pred = logisticRegr.predict(new_test_df)

ids = titanic_test_df['PassengerId']

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': y_pred })

output.to_csv('titanic-predictions.csv', index = False)

output.head()