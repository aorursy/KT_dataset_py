# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



#sklearn 



from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix



import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="darkgrid")



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
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

display(df_test.head())

display(df_train.head())
sns.countplot(x='Survived',data = df_train)
#Dropping passenger id, name, ticket



df_train.drop(['PassengerId', 'Name','Fare','Ticket'], axis=1, inplace=True)

df_test.drop(['PassengerId', 'Name','Fare','Ticket'], axis=1, inplace=True)

target = df_train.pop('Survived')
df_train.replace({"Sex":{"male":1,

                  "female":2}} ,inplace=True)

df_train["Embarked"] = df_train["Embarked"].fillna('S')
df_test.replace({"Sex":{"male":1,

                  "female":2}} ,inplace=True)

df_test["Embarked"] = df_train["Embarked"].fillna('S')
df_train.isnull().sum()
df_test.isnull().sum()
df_train = df_train.interpolate()

df_test= df_test.interpolate()
df_train.isnull().sum()
df_test.isnull().sum()
df_train.Cabin.unique()
df_train.drop('Cabin', axis=1, inplace=True)

df_test.drop('Cabin',axis=1,inplace=True)
df_train = pd.get_dummies(df_train, columns=["Sex", "Embarked","Pclass"], prefix=["sex", "embarked","Pclass"])

df_test = pd.get_dummies(df_test, columns=["Sex", "Embarked",'Pclass'], prefix=["sex", "embarked","Pclass"])
print(df_train.shape,target.shape)

print(df_test.shape)
X_train, X_test, y_train, y_test = train_test_split(df_train, target, test_size=0.33, random_state=42)
clf = RandomForestClassifier(random_state=0)



clf.fit(X_train,y_train)



y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

print(accuracy*100)
confusion_matrix(y_test,y_pred)
import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)

plt.show()
prediction = clf.predict(df_test)
test = pd.read_csv('/kaggle/input/titanic/test.csv')

test.head()
predicted_values = prediction.astype(int)

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predicted_values

    })

submission.to_csv('titanic_submission.csv',index=False)

display(submission.tail())



print("The shape of the sumbission file is {}".format(submission.shape))