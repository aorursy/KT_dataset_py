import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_val_score

from sklearn import metrics



from sklearn import preprocessing



from yellowbrick.classifier import ConfusionMatrix

from yellowbrick.classifier import ClassificationReport

plt.style.use('ggplot')
dataset = pd.read_csv('../input/train.csv')

dataset.head()
dataset.describe(include = "all")
dataset.shape
dataset.isnull().sum(axis=0)
sns.countplot(dataset['Embarked'])
# File missing values in embarked with S which is the most frequent item.

dataset = dataset.fillna({"Embarked": "S"})
# Sex feature

le_sex = preprocessing.LabelEncoder()

le_sex.fit(["male", "female"])

dataset['Sex'] = le_sex.transform(dataset['Sex'])
# Embark feature

le_embark = preprocessing.LabelEncoder()

le_embark.fit(["S", "C", "Q"])

dataset['Embarked'] = le_embark.transform(dataset['Embarked'])
train_class = dataset[['Survived']]

train_feature = dataset[['Pclass', 'Sex', 'Embarked', 'Parch', 'SibSp', 'Fare']]

train_feature.head()
clf = GaussianNB()



scoring = {'acc': 'accuracy',

           'prec_macro': 'precision_macro',

           'rec_macro': 'recall_macro',

           'f1_macro': 'f1_macro'}

scores = cross_validate(clf, train_feature, train_class.values.ravel(), cv=10, scoring=scoring)



print(scores.keys())



print ('Accuracy score : %.3f' % scores['test_acc'].mean())

print ('Precisoin score : %.3f' % scores['test_prec_macro'].mean())

print ('Recall score : %.3f' % scores['test_rec_macro'].mean())

print ('F1 score : %.3f' % scores['test_f1_macro'].mean())
# Loading test dataset

test = pd.read_csv('../input/test.csv')



# Fit the model

clf.fit(train_feature, train_class.values.ravel())



# Replace missing Fare values with mean

meanFare = dataset['Fare'].mean()

test = test.fillna({"Fare": meanFare})



# Categorical -> numerical conversion

test['Sex'] = le_sex.transform(test['Sex'])

test['Embarked'] = le_embark.transform(test['Embarked'])



#set ids as PassengerId and predict survival

ids = test['PassengerId']

test_feature = test[['Pclass', 'Sex', 'Embarked', 'Parch', 'SibSp', 'Fare']]

predictions = clf.predict(test_feature)



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.head()
output.to_csv('submission.csv', index=False)