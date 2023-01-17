import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

import seaborn as sns
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

submion = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.columns
sns.heatmap(train.isnull(), yticklabels=False, cmap="viridis")
# Fillna with average age

train['Age'] = train['Age'].fillna(train['Age'].mean())

test['Age'] = test['Age'].fillna(train['Age'].mean())



# Fillna with most frequent cabin

train['Cabin'] = train['Cabin'].fillna(train['Cabin'].value_counts()[:1].index.tolist()[0])

test['Cabin'] = test['Cabin'].fillna(train['Cabin'].value_counts()[:1].index.tolist()[0])



sns.heatmap(train.isnull(), yticklabels=False, cmap="viridis")
# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()

# cabin = le.fit_transform(train.Cabin.dropna())

# sns.distplot(cabin)
sns.set_style('whitegrid')
sns.countplot(x = "Survived", data=train, hue="Sex")
sns.countplot(x = "Survived", data=train, hue="Pclass")
sns.countplot(x = "Survived", data=train, hue="Parch")
sns.countplot(x = "Survived", data=train, hue="Embarked")
train['Cabin'].value_counts()[:1].index.tolist()[0]
sns.distplot(train.Age)
from sklearn import svm





cat_features = ["Parch", "Sex", "Pclass", 'Age', "Embarked", "Survived"]

train = train[cat_features]

train_X = train.drop("Survived", axis=1)

train_Y = train["Survived"]



cat_features = ["Parch", "Sex", "Pclass", 'Age', "Embarked"]

test = test[cat_features]

# test_X = test.drop("Survived", axis=1)

# test = test["Survived"]



one_hot_encoded_training_predictors = pd.get_dummies(train_X)

one_hot_encoded_training_predictors.head()



# creating model

clf=svm.SVC(gamma="auto")

clf.fit(one_hot_encoded_training_predictors,train_Y)



#Import Random Forest Model

# from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier

# clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)

# clf.fit(one_hot_encoded_training_predictors,train_Y)



one_hot_encoded_training_predictors_test = pd.get_dummies(test)



predict=clf.predict(one_hot_encoded_training_predictors_test)
test_df = pd.read_csv('../input/titanic/test.csv')

submission = pd.DataFrame({

    'PassengerId': test_df['PassengerId'],

    'Survived': predict

})

submission.to_csv('titanic_prediction.csv', index=False)