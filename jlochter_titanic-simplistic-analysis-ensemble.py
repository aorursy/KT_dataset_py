import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv('../input/train.csv')
test_df  = pd.read_csv('../input/test.csv')

print('Shape for training and testing dataset')
print('Train', train_df.shape)
print('Test', test_df.shape)

print()
print('Show which field has missing values in training dataset')
print(train_df.isna().sum())

print()
print('Show which field has missing values in testing dataset')
print(test_df.isna().sum())

## create a title field
import re

rare_title = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']

titles = train_df.Name.tolist()
titles = [re.sub('(.*, )|(\\..*)', '', t) for t in titles]
train_df['Title'] = titles

titles = test_df.Name.tolist()
titles = [re.sub('(.*, )|(\\..*)', '', t) for t in titles]
test_df['Title'] = titles

train_df.loc[train_df.Title == 'Mlle', 'Title'] = 'Miss'
train_df.loc[train_df.Title == 'Ms', 'Title'] = 'Miss'
train_df.loc[train_df.Title == 'Mme', 'Title'] = 'Mrs'
train_df.loc[train_df.Title.isin(rare_title), 'Title'] = 'Rare'

test_df.loc[test_df.Title == 'Mlle', 'Title'] = 'Miss'
test_df.loc[test_df.Title == 'Ms', 'Title'] = 'Miss'
test_df.loc[test_df.Title == 'Mme', 'Title'] = 'Mrs'
test_df.loc[test_df.Title.isin(rare_title), 'Title'] = 'Rare'

## To fix missing values in a lazy manner, drop those fields

train_df.drop(columns=['Age','Name','Ticket','Cabin'], inplace=True)
test_df.drop(columns=['Age','Name','Ticket','Cabin'], inplace=True)

print('Shape for training and testing dataset')
print('Train', train_df.shape)
print('Test', test_df.shape)
print()

## But we still need to fill those Embarked in training dataset and that fare one in testing dataset
print('The uniques values for Embarked field:')
print(train_df.Embarked.unique())
train_df.Embarked.fillna('C', inplace=True)

print()
print('The fare average for Fare field in testing dataset:')
fare_mean = test_df.Fare.mean()
print(fare_mean)
test_df.Fare.fillna(fare_mean, inplace=True)

print()
print('Show which field has missing values in training dataset')
print(train_df.isna().sum())

print()
print('Show which field has missing values in testing dataset')
print(test_df.isna().sum())
## Converting fields to categorical and show its correlation

train_X = pd.get_dummies(data=train_df,columns=['Title','Pclass','Parch','Sex','SibSp','Embarked'])
test_X  = pd.get_dummies(data=test_df, columns=['Title','Pclass','Parch','Sex','SibSp','Embarked'])

train_Y = train_X.Survived
test_PassengerId = test_X.PassengerId

train_X.drop(columns=['PassengerId'], inplace=True)
test_X.drop(columns=['PassengerId'], inplace=True)

corr = train_X.corr()
f, ax = plt.subplots(figsize=(10, 10)) 
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=1.0, square=True, linewidths=.3, cbar_kws={"shrink": .5}, ax=ax) 
plt.show()

train_X.drop(columns=['Survived'], inplace=True)

## after dummies, check for shapes and fill columns to match them
print('Shapes:')
print(train_X.shape)
print(test_X.shape)

print()
print('Show columns for each dataset:')
print('Train:', sorted(train_X.columns))
print('Test:', sorted(test_X.columns))

train_X['Parch_9'] = 0

print('Shapes:')
print(train_X.shape)
print(test_X.shape)

## do machine learning magick 
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = MultinomialNB()
clf4 = SVC(kernel='rbf', probability=True)

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('svm', clf4)], voting='hard')

for clf, label in zip([clf1, clf2, clf3, clf4, eclf], ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'SVM', 'Ensemble']):
    scores = cross_val_score(clf, train_X, train_Y, cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    
eclf.fit(train_X, train_Y)
eclf_pred = eclf.predict(test_X)
## generates output

submission = pd.DataFrame(
    {'PassengerId': test_PassengerId, 'Survived': eclf_pred},
    columns = ['PassengerId', 'Survived'])
submission.to_csv('submission.csv', index = False)

print(os.listdir('.'))
