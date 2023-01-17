#Linear Algebra
import numpy as np
# Loading + Cleaning
import pandas as pd
pd.set_option('max_rows',5)
# Visualization
import seaborn as sns
# ML
from sklearn import linear_model
# We start by loading the data
train_d = pd.read_csv('../input/train.csv')
train_d = pd.DataFrame(train_d)
test_d = pd.read_csv('../input/test.csv')
test_d = pd.DataFrame(test_d)
train_d.head()
# We will only need Sex , Age, Survived columns
train_d = train_d.drop(columns={'Name','Ticket','Fare','Cabin','Embarked','Parch','SibSp'})
train_d.isnull().sum()
# The age columns is missing 177 values let's fill them with the mean
train_d.Age = train_d.Age.fillna(train_d.Age.mean())
sns.boxplot(x='Pclass',y='Age',data=train_d,palette='winter')
train_features = train_d.drop(['Survived','Pclass','PassengerId'],axis=1)
train_features['S'] = train_d['Sex'].map({'male':1,'female':0})
train_features =  train_features.drop(columns={'Sex'})
x_features = train_features.as_matrix()
y_label = train_d['Survived'].as_matrix()
# Load the model
lg = linear_model.LogisticRegression(C=1)
lg.fit(x_features,y_label)
# Get the features
test_features = test_d.drop(columns={'PassengerId','Pclass','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'})
test_features
test_features['Age'] = test_d['Age'].fillna(test_d.Age.mean())
test_features
# Map categorical data to numerical
test_features['S'] = test_features.Sex.map({'male':1,'female':0})
test_features = test_features.drop(columns={'Sex'})
test_features = test_features.as_matrix()
pred = lg.predict(test_features)
Survived = pd.DataFrame(data=pred,columns=['Survived'])
Survived.to_csv('test.csv')