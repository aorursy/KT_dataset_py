# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns # data visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.



test_data = pd.read_csv("../input/test.csv")

train_data = pd.read_csv("../input/train.csv")

train_data.rename(columns={'SibSp':'SiblingsSpouses', 'Parch':'ParentsChildren'}, inplace = True)

test_data.rename(columns={'SibSp':'SiblingsSpouses', 'Parch':'ParentsChildren'}, inplace = True)

test_columns = test_data.columns

train_columns = train_data.columns
print ("\nNaN rows per column in training data\n")

for col in train_columns:

    print (col, ": ", train_data[train_data[col].isnull()].shape[0])

    

print ("\nNaN rows per column in test data\n")

for col in test_columns:

    print (col, ": ", test_data[test_data[col].isnull()].shape[0])
temp = train_data[pd.notnull(train_data['Age'])]

maletemp = temp[temp['Sex'] == 'male']

femaletemp = temp[temp['Sex'] == 'female']

youngmisstemp = femaletemp[femaletemp['Name'].str.contains('Miss')]

youngmisstemp = youngmisstemp[youngmisstemp['ParentsChildren'] > 0]

oldmisstemp = femaletemp[femaletemp['Name'].str.contains('Miss')]

oldmisstemp = oldmisstemp[oldmisstemp['ParentsChildren'] == 0]



mastermedian = maletemp[maletemp['Name'].str.contains('Master')]['Age'].median()

mistermedian = maletemp[maletemp['Name'].str.contains('Mr.')]['Age'].median()

mrsmedian = femaletemp[femaletemp['Name'].str.contains('Mrs.')]['Age'].median()

youngmissmedian = youngmisstemp['Age'].median()

oldmissmedian = oldmisstemp['Age'].median()
print("MasterMedian", mastermedian)

print("MisterMedian", mistermedian)

print("MrsMedian", mrsmedian)

print("OldMissMedian", oldmissmedian)

print("YoungMissMedian", youngmissmedian)
mastermask = (train_data['Name'].str.contains('Master')) & (train_data['Sex'] == 'male') & (np.isnan(train_data['Age']))

mrmask = (train_data['Name'].str.contains('Mr.')) & (train_data['Sex'] == 'male') & (np.isnan(train_data['Age']))

mrsmask = (train_data['Name'].str.contains('Mrs.')) & (train_data['Sex'] == 'female') & (np.isnan(train_data['Age']))

youngmissmask = (train_data['Name'].str.contains('Miss')) & (train_data['Sex'] == 'female') & (train_data['ParentsChildren']>0) & (np.isnan(train_data['Age']))

oldmissmask = (train_data['Name'].str.contains('Miss')) & (train_data['Sex'] == 'female') & (train_data['ParentsChildren']==0) & (np.isnan(train_data['Age']))
train_data.loc[mastermask, 'Age'] = 3.5

train_data.loc[mrmask, 'Age'] = 30

train_data.loc[mrsmask, 'Age'] = 35

train_data.loc[youngmissmask, 'Age'] = 9

train_data.loc[oldmissmask, 'Age'] = 26
print (train_data[pd.isnull(train_data['Age'])])
train_data.loc[train_data['PassengerId'] == 767, 'Age'] = 30

print (train_data[pd.isnull(train_data['Age'])])
temp = test_data



mastermask = (temp['Name'].str.contains('Master')) & (temp['Sex'] == 'male') & (np.isnan(temp['Age']))

mrmask = (temp['Name'].str.contains('Mr.')) & (temp['Sex'] == 'male') & (np.isnan(temp['Age']))

mrsmask = (temp['Name'].str.contains('Mrs.')) & (temp['Sex'] == 'female') & (np.isnan(temp['Age']))

youngmissmask = (temp['Name'].str.contains('Miss')) & (temp['Sex'] == 'female') & (temp['ParentsChildren']>0) & (np.isnan(temp['Age']))

oldmissmask = (temp['Name'].str.contains('Miss')) & (temp['Sex'] == 'female') & (temp['ParentsChildren']==0) & (np.isnan(temp['Age']))

oldmissmask2 = (temp['Name'].str.contains('Ms.')) & (temp['Sex'] == 'female') & (temp['ParentsChildren']==0) & (np.isnan(temp['Age']))



temp.loc[mastermask, 'Age'] = 3.5

temp.loc[mrmask, 'Age'] = 30

temp.loc[mrsmask, 'Age'] = 35

temp.loc[youngmissmask, 'Age'] = 9

temp.loc[oldmissmask, 'Age'] = 26

temp.loc[oldmissmask2, 'Age'] = 26
print (temp[pd.isnull(temp['Age'])])
test_data_clean = temp
print ("Pclass v Survived")
temp = train_data[['Survived','Pclass']]

plt.figure()

sns.countplot(x='Pclass', hue='Survived', data=temp)

plt.figure()

sns.barplot(x='Pclass', y='Survived', data=temp)
print ("Sex v Survived")
temp = train_data[['Survived','Sex']]

plt.figure()

sns.countplot(x='Sex', hue='Survived', data=temp)

plt.figure()

sns.barplot(x='Sex', y='Survived', data=temp)
print ("Age v Survived")
temp = train_data[['Survived','Age']]



plt.figure()

plt.title('Non survivors')

sns.distplot(temp['Age'][(temp['Survived'] == 0)])

plt.figure()

plt.title('Survivors')

sns.distplot(temp['Age'][(temp['Survived'] == 1)])
temp = train_data[['Age', 'Survived', 'Sex']]

female_temp = temp[temp['Sex'] == 'female']

male_temp = temp[temp['Sex'] == 'male']



female_temp['AgeBucket'] = female_temp['Age'].apply(lambda x: x//10)

plt.figure()

plt.title('Females')

sns.countplot(x='AgeBucket', hue='Survived', data = female_temp)

plt.figure()

plt.title('Females')

sns.barplot(x='AgeBucket', y='Survived', data = female_temp)



male_temp['AgeBucket'] = male_temp['Age'].apply(lambda x: x//10)

plt.figure()

plt.title('Males')

sns.countplot(x='AgeBucket', hue='Survived', data = male_temp)

plt.figure()

plt.title('Males')

sns.barplot(x='AgeBucket', y='Survived', data = male_temp)
print ("SiblingsSpouses v Survived")
temp = train_data[['Survived','SiblingsSpouses']]

temp['SiblingsSpouses'] = temp['SiblingsSpouses'].apply(lambda x: 1 if x > 0 else 0)



plt.figure()

sns.countplot(x='SiblingsSpouses', hue='Survived', data=temp)

plt.figure()

sns.barplot(x='SiblingsSpouses', y='Survived', data=temp)
print ("ParentsChildren v Survived")
temp = train_data[['Survived','ParentsChildren']]

temp['ParentsChildren'] = temp['ParentsChildren'].apply(lambda x: 1 if x > 0 else 0)



plt.figure()

sns.countplot(x='ParentsChildren', hue='Survived', data=temp)

plt.figure()

sns.barplot(x='ParentsChildren', y='Survived', data=temp)
print ("Embarked v Survived")
temp = train_data[['Survived','Embarked']]



plt.figure()

sns.countplot(x='Embarked', hue='Survived', data=temp)

plt.figure()

sns.barplot(x='Embarked', y='Survived', data=temp)
temp = train_data[['Embarked', 'Sex', 'Survived']]

plt.figure()

sns.countplot(x='Embarked', hue = 'Sex', data=temp)
temp = train_data[~(train_data['Name'].str.contains('Mr.') | train_data['Name'].str.contains('Master') | \

                  train_data['Name'].str.contains('Mrs.') | train_data['Name'].str.contains('Miss') | \

                  train_data['Name'].str.contains('Ms.'))]

print (temp['Name'])
print (temp[temp['Survived'] == 1])
train_data_features = pd.concat([train_data['Age'], train_data['Sex'], train_data['Pclass'], \

                                 train_data['SiblingsSpouses'], train_data['ParentsChildren'], \

                                train_data['Name']], axis=1)

     

train_data_features['SiblingsSpouses'] = train_data_features['SiblingsSpouses'].apply(lambda x: 1 if x > 0 else 0)

train_data_features['ParentsChildren'] = train_data_features['ParentsChildren'].apply(lambda x: 1 if x > 0 else 0)

train_data_features['Sex'] = train_data_features['Sex'].apply(lambda x: 1 if x =='female' else 0)



train_data_features['Name'] = train_data_features['Name'].apply(lambda x: 0 if ('Mr.' in x or 'Master' in x or 'Mrs.' in x \

                                                                               or 'Miss' in x or 'Ms.' in x) else 1)
from sklearn import tree

basic_model = tree.DecisionTreeClassifier()
y_true = train_data['Survived']

from sklearn.model_selection import KFold

splits = 10

kf = KFold(n_splits = splits, shuffle = True)

accuracy = 0

for train_fold, cv_fold in kf.split(train_data_features):

    basic_model.fit(train_data_features.loc[train_fold], train_data.loc[train_fold,'Survived'])

    y_true = train_data.loc[cv_fold, 'Survived']

    accuracy = accuracy + basic_model.score(train_data_features.loc[cv_fold], y_true)



accuracy = accuracy/splits



print ("Basic Decision Tree accuracy: ", accuracy)
from sklearn import ensemble

adaboostclassifier = ensemble.AdaBoostClassifier()
splits = 10

kf = KFold(n_splits = splits, shuffle = True)

accuracy = 0

for train_fold, cv_fold in kf.split(train_data_features):    

    adaboostclassifier.fit(train_data_features.loc[train_fold], train_data.loc[train_fold,'Survived'])

    y_true = train_data.loc[cv_fold, 'Survived']

    accuracy = accuracy + adaboostclassifier.score(train_data_features.loc[cv_fold], y_true)

    

accuracy = accuracy/10

print ("Adaboost Decision tree accuracy: ", accuracy)
splits = 10

kf = KFold(n_splits = splits, shuffle = True)

randomforestclassifier = ensemble.RandomForestClassifier()

accuracy = 0

for train_fold, cv_fold in kf.split(train_data_features): 

    randomforestclassifier.fit(train_data_features.loc[train_fold], train_data.loc[train_fold,'Survived'])

    y_true = train_data.loc[cv_fold, 'Survived']

    accuracy = accuracy + randomforestclassifier.score(train_data_features.loc[cv_fold], y_true)

    

accuracy = accuracy/10

print ("Random Forests accuracy: ", accuracy)
splits = 10

kf = KFold(n_splits = 10, shuffle = True)

accuracy = 0

max_accuracy = 0

best_estimators = 0

total_estimators = [10,20,30,40,50,60,70,80,90,100]

for estimators in total_estimators:

    adaboostclassifier = ensemble.AdaBoostClassifier(n_estimators=estimators)

    for train_fold, cv_fold in kf.split(train_data_features):    

        adaboostclassifier.fit(train_data_features.loc[train_fold], train_data.loc[train_fold,'Survived'])

        y_true = train_data.loc[cv_fold, 'Survived']

        accuracy = accuracy + adaboostclassifier.score(train_data_features.loc[cv_fold], y_true)    

    accuracy = accuracy/10

    if (accuracy > max_accuracy):

        max_accuracy = accuracy

        best_estimators = estimators
print ("Adaboost Decision tree max accuracy: ", max_accuracy, "at", best_estimators, "estimators.")
splits = 10

kf = KFold(n_splits = 10, shuffle = True)

accuracy = 0

max_accuracy = 0

best_depth = 0

depthrange = range(1, train_data_features.shape[1])

for depth in depthrange:

    basic_model = tree.DecisionTreeClassifier(max_depth = depth)

    for train_fold, cv_fold in kf.split(train_data_features):    

        basic_model.fit(train_data_features.loc[train_fold], train_data.loc[train_fold,'Survived'])

        y_true = train_data.loc[cv_fold, 'Survived']

        accuracy = accuracy + basic_model.score(train_data_features.loc[cv_fold], y_true)    

    accuracy = accuracy/10

    if (accuracy > max_accuracy):

        max_accuracy = accuracy

        best_depth = depth

    

print ("Basic Decision tree accuracy: ", max_accuracy, "at", best_depth, "depth.")
test_data_clean_features = pd.concat([test_data_clean['Age'], test_data_clean['Sex'], test_data_clean['Pclass'], \

                                 test_data_clean['SiblingsSpouses'], test_data_clean['ParentsChildren'], \

                                test_data_clean['Name']], axis=1)

PID = test_data_clean['PassengerId']



test_data_clean_features['SiblingsSpouses'] = test_data_clean_features['SiblingsSpouses'].apply(lambda x: 1 if x > 0 else 0)

test_data_clean_features['ParentsChildren'] = test_data_clean_features['ParentsChildren'].apply(lambda x: 1 if x > 0 else 0)

test_data_clean_features['Sex'] = test_data_clean_features['Sex'].apply(lambda x: 1 if x =='female' else 0)



test_data_clean_features['Name'] = test_data_clean_features['Name'].apply(lambda x: 0 if ('Mr.' in x or 'Master' in x \

                                                                                          or 'Mrs.' in x or 'Miss' in x \

                                                                                          or 'Ms.' in x) else 1)
basic_model = tree.DecisionTreeClassifier(max_depth = 5)

basic_model.fit(train_data_features, train_data['Survived'])

test_predictions = basic_model.predict(test_data_clean_features)

submission = pd.DataFrame({"PassengerId" : PID, "Survived" : test_predictions})
submission.to_csv("submission.csv", index=False)