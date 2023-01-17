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



df = pd.read_csv('../input/titanic/train.csv')

df.reset_index(drop = True)

df
dummy_sex = pd.get_dummies(df["Sex"])

dummy_embarked = pd.get_dummies(df["Embarked"])



new_data = pd.concat([df,dummy_sex,dummy_embarked], axis=1)



new_data = new_data.rename(columns={'female':'isFemale'})

new_data
new_data.drop(['Name' , 'Ticket' , 'Cabin' , 'PassengerId'  , 'Embarked' , 'SibSp' , 'Sex' , 'male' ] ,axis = 1)
new_data.corr()
from sklearn.svm import LinearSVC

from sklearn.neighbors import KNeighborsClassifier
X = new_data[['Pclass','Fare','isFemale','C','S']]

Y = new_data.Survived



clf = KNeighborsClassifier()

clf.fit(X, Y)
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.count()
test_data.head()
test_data.Fare = test_data.Fare.fillna(test_data.Fare.mean())



dummy_sex_test = pd.get_dummies(test_data["Sex"])

dummy_embarked_test = pd.get_dummies(test_data["Embarked"])



new_test_data = pd.concat([test_data,dummy_sex_test,dummy_embarked_test], axis=1)



new_test_data = new_test_data.rename(columns={'female':'isFemale'})



X_test = new_test_data[['Pclass','Fare','isFemale','C','S']]
predicted_data = clf.predict(X_test)
predicted_data
final = pd.DataFrame(

                            {

                                'PassengerId':test_data.PassengerId,

                                'Survived':predicted_data

                            },

                            columns=['PassengerId','Survived']

                        )

final.to_csv('my_submission.csv', index=False)