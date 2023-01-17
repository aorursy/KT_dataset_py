# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")

#train_df = train_df.dropna(axis=1)

#train_df['Age'].dropna().astype(int)









test_df = pd.read_csv("../input/test.csv")

test_y = pd.read_csv("../input/gender_submission.csv")



X_train = train_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

sex = pd.get_dummies(X_train["Sex"])

X_train = X_train.drop(["Sex"], axis = 1)

X_train = X_train.join(sex)

embarked = pd.get_dummies(X_train["Embarked"])

X_train = X_train.drop(["Embarked"], axis = 1)

X_train = X_train.join(embarked)

Y_train = train_df.drop(["PassengerId", "Ticket", "Cabin", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Name"], axis = 1 )







X_test = test_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

sex = pd.get_dummies(X_test["Sex"])

X_test = X_test.drop(["Sex"], axis = 1)

X_test = X_test.join(sex)

embarked = pd.get_dummies(X_test["Embarked"])

X_test = X_test.drop(["Embarked"], axis = 1)

X_test = X_test.join(embarked)

Y_test = test_y.drop("PassengerId", axis=1).copy()



X_train.head()

train_df.info()

train_df.describe()
class_1_pass = train_df[train_df["Pclass"]==1]

print('1st Class Passengers :')

print(class_1_pass['Fare'].count())



class_1_pass_sur = class_1_pass[class_1_pass["Survived"]==1]

print('Survivor of 1st class passengers: ')

print(class_1_pass_sur['Fare'].count())





class_1_pass_female = class_1_pass[class_1_pass["Sex"]=="female"]

print('No of female in 1st class : ')

print(class_1_pass_female["PassengerId"].count())







class_1_pass_sur_female = class_1_pass_sur[class_1_pass_sur["Sex"]=="female"]

print('Female survivor of 1st class passengers :')

print(class_1_pass_sur_female['PassengerId'].count())



class_1_female_sur_rate = class_1_pass_sur_female['PassengerId'].count()/class_1_pass_female['PassengerId'].count()

print('1st class female survivor rate :')

print(class_1_female_sur_rate)



class_1_pass_child = class_1_pass[class_1_pass["Age"]<15]

print('No of childrn in 1st class : ')

print(class_1_pass_child["PassengerId"].count())



class_1_pass_sur_child = class_1_pass_sur[class_1_pass_sur["Age"]<15]

print('Children survivor of 1st class passengers')

print(class_1_pass_sur_child['PassengerId'].count())



class_1_child_sur_rate = class_1_pass_sur_child['PassengerId'].count()/class_1_pass_child['PassengerId'].count()

print('1st class child survivor rate :')

print(class_1_child_sur_rate)





class_2_pass = train_df[train_df["Pclass"]==2]

print('2nd Class Passengers :')

print(class_2_pass['Fare'].count())



class_2_pass_sur = class_2_pass[class_2_pass["Survived"]==1]

print('Survivor of 2nd class passengers: ')

print(class_2_pass_sur['Fare'].count())





class_2_pass_female = class_2_pass[class_2_pass["Sex"]=="female"]

print('No of female in 2nd class : ')

print(class_2_pass_female["PassengerId"].count())







class_2_pass_sur_female = class_2_pass_sur[class_2_pass_sur["Sex"]=="female"]

print('Female survivor of 2nd class passengers :')

print(class_2_pass_sur_female['PassengerId'].count())



class_2_female_sur_rate = class_2_pass_sur_female['PassengerId'].count()/class_2_pass_female['PassengerId'].count()

print('2nd class female survivor rate :')

print(class_2_female_sur_rate)



class_2_pass_child = class_2_pass[class_2_pass["Age"]<15]

print('No of childrn in 2nd class : ')

print(class_2_pass_child["PassengerId"].count())



class_2_pass_sur_child = class_2_pass_sur[class_2_pass_sur["Age"]<15]

print('Children survivor of 2nd class passengers')

print(class_2_pass_sur_child['PassengerId'].count())



class_2_child_sur_rate = class_2_pass_sur_child['PassengerId'].count()/class_2_pass_child['PassengerId'].count()

print('2nd class child survivor rate :')

print(class_2_child_sur_rate)





class_3_pass = train_df[train_df["Pclass"]==3]

print('3rd Class Passengers :')

print(class_3_pass['PassengerId'].count())



class_3_pass_sur = class_3_pass[class_3_pass["Survived"]==1]

print('Survivor of 3rd class passengers: ')

print(class_3_pass_sur['PassengerId'].count())





class_3_pass_female = class_3_pass[class_3_pass["Sex"]=="female"]

print('No of female in 3rd class : ')

print(class_3_pass_female["PassengerId"].count())







class_3_pass_sur_female = class_3_pass_sur[class_3_pass_sur["Sex"]=="female"]

print('Female survivor of 3rd class passengers :')

print(class_3_pass_sur_female['PassengerId'].count())



class_3_female_sur_rate = class_3_pass_sur_female['PassengerId'].count()/class_3_pass_female['PassengerId'].count()

print('3rd class female survivor rate :')

print(class_3_female_sur_rate)



class_3_pass_child = class_3_pass[class_3_pass["Age"]<15]

print('No of childrn in 3rd class : ')

print(class_3_pass_child["PassengerId"].count())



class_3_pass_sur_child = class_3_pass_sur[class_3_pass_sur["Age"]<15]

print('Children survivor of 3rd class passengers')

print(class_3_pass_sur_child['PassengerId'].count())



class_3_child_sur_rate = class_3_pass_sur_child['PassengerId'].count()/class_3_pass_child['PassengerId'].count()

print('3rd class child survivor rate :')

print(class_3_child_sur_rate)





## Applying logistic regression on the data set



logreg = LogisticRegression()

#X_train = pd.get_dummies(X_train)

#X_test = pd.get_dummies(X_test)



from sklearn.preprocessing import Imputer

my_imputer = Imputer()

X_train = my_imputer.fit_transform(X_train)

X_test = my_imputer.fit_transform(X_test)



logreg.fit(X_train, Y_train.values.ravel())



Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)