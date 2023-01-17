import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print(train.shape, train.columns.values)

print(train.isnull().sum())

train.head(3)

#select the features and separate the target



features = ['Pclass','Sex','SibSp','Parch','Fare','Cabin','Embarked']

X_train= train[features]

X_test=test[features]

y_train=train['Survived']



# do all the replacements and transformations as described in the introduction



for Z in [X_train, X_test]:

    

    A = Z.groupby(['Pclass'])['Fare'].mean()

    B = Z['Pclass'].map({1: A.iloc[0], 2:A.iloc[1], 3:A.iloc[2]})

    Z['Fare'] = Z['Fare'].fillna(B)

    Z['Embarked'] = Z['Embarked'].fillna('S')

    Z['Cabin'] = Z['Cabin'].fillna(1)

    Z['Sex'] = Z['Sex'].map({'male': 0, 'female':1}).astype(int)

    Z['Embarked'] = Z['Embarked'].map({'C': 0, 'Q':1, 'S':2}).astype(int)

    Z.loc[Z['Cabin'] != 1, 'Cabin'] = 0

    Z['Cabin'] = pd.to_numeric(Z['Cabin'], errors='coerce')

    

# double check the dataset that it looks reasonable    

X_train.describe()    

#use Gradient Boosting with default paramenetrs to train the model and submit the prediction



from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier().fit(X_train, y_train)

print('Accuracy on the training set: {:.2f}' .format(clf.score(X_train, y_train)))



prediction = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": clf.predict(X_test)})

prediction.to_csv('my_submission.csv', index=False)
