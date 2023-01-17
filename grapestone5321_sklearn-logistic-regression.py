import numpy as np 

import pandas as pd 



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



all = pd.concat([train, test], sort = False)

all['Age'] = all['Age'].fillna(value=all['Age'].median())

all['Fare'] = all['Fare'].fillna(value=all['Fare'].median())

all['Embarked'] = all['Embarked'].fillna('S')



all.loc[ all['Age'] <= 16, 'Age'] = 0

all.loc[(all['Age'] > 16) & (all['Age'] <= 32), 'Age'] = 1

all.loc[(all['Age'] > 32) & (all['Age'] <= 48), 'Age'] = 2

all.loc[(all['Age'] > 48) & (all['Age'] <= 64), 'Age'] = 3

all.loc[ all['Age'] > 64, 'Age'] = 4



import re

def get_title(name):

    title_search = re.search(' ([A-Za-z]+\.)', name)

    

    if title_search:

        return title_search.group(1)

    return ""



all['Title'] = all['Name'].apply(get_title)

all['Title'].value_counts()



all['Title'] = all['Title'].replace(['Capt.', 'Dr.', 'Major.', 'Rev.'], 'Officer.')

all['Title'] = all['Title'].replace(['Lady.', 'Countess.', 'Don.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Royal.')

all['Title'] = all['Title'].replace(['Mlle.', 'Ms.'], 'Miss.')

all['Title'] = all['Title'].replace(['Mme.'], 'Mrs.')

all['Title'].value_counts()



all['Cabin'] = all['Cabin'].fillna('Missing')

all['Cabin'] = all['Cabin'].str[0]

all['Cabin'].value_counts()



all['Family_Size'] = all['SibSp'] + all['Parch'] + 1

all['IsAlone'] = 0

all.loc[all['Family_Size']==1, 'IsAlone'] = 1



all_1 = all.drop(['Name', 'Ticket'], axis = 1)

all_dummies = pd.get_dummies(all_1, drop_first = True)

all_train = all_dummies[all_dummies['Survived'].notna()]

all_test = all_dummies[all_dummies['Survived'].isna()]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(all_train.drop(['PassengerId','Survived'],axis=1), 

                                                    all_train['Survived'], 

                                                    test_size=0.30, 

                                                    random_state=111)
from sklearn.linear_model import LogisticRegression



model = LogisticRegression(solver = 'liblinear')

model.fit(X_train,y_train)

predictions = model.predict(X_test)



from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,predictions)



TestForPred = all_test.drop(['PassengerId', 'Survived'], axis = 1)

t_pred = model.predict(TestForPred).astype(int)

PassengerId = all_test['PassengerId']



sub = pd.DataFrame({'PassengerId': PassengerId, 'Survived':t_pred })

sub.to_csv("Submission.csv", index = False)