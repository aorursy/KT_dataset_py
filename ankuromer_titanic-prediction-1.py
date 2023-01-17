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
import numpy as np

import pandas as pd



train = pd.read_csv(r'/kaggle/input/titanic/train.csv')

test = pd.read_csv(r'/kaggle/input/titanic/test.csv')

result = pd.read_csv(r'/kaggle/input/titanic/gender_submission.csv')



survived = train[train['Survived'] == 1]

non_survived = train[train['Survived'] == 0]

print ("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(train)*100.0))

print("not survived: %i (%.1f%%)"%(len(non_survived), float(len(non_survived))/len(train)*100.0))



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}



train_test_data = [train, test]

for dataset in train_test_data:

    dataset['Title'] = dataset.Name.str.extract( '([A-Za-z]+)\.')

    dataset['Title'] = dataset['Title'].replace(['Lady','Col','Rev','Dr','Major', 'Capt','Jonkheer','Countess', 'Don','Sir'], 'Other')

    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'],'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    dataset['Sex'] = dataset['Sex'].map({"female": 1, "male": 0}).astype(int)

    

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null = np.random.randint(age_avg - age_std, age_avg + age_std, size = age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null.copy()

    dataset['Age'] = dataset['Age'].astype(int) 

    

train['AgeRange'] = pd.cut(train['Age'],5)

train.AgeRange.value_counts()



for dataset in train_test_data:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

    

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)



for dataset in train_test_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['FareBand'] = pd.qcut(train['Fare'], 4)

print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())  

##train['FareRange'] =  pd.cut(train['Fare'],4)  

##train.FareRange.value_counts()



features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin']

train = train.drop(features_drop, axis=1)

test = test.drop(features_drop, axis=1)

train = train.drop(['PassengerId', 'AgeRange', 'FareBand'], axis=1)



X_train = train.drop('Survived', axis = 1)

Y_train = train['Survived']

X_test = test.drop("PassengerId", axis=1).copy()

Y_test = result['Survived']

###print(X_test)



from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV

lr = DecisionTreeClassifier()



cv_scores = cross_val_score(lr, X_train, Y_train, cv = 5)

cv_scores.mean()



parameters = {'max_depth':[1,2,3,4,5], 

              'min_samples_leaf':[1,2,3,4,5], 

              'min_samples_split':[2,3,4,5],

              'criterion' : ['gini','entropy']}



grid_obj = GridSearchCV(lr, parameters)

grid_fit = grid_obj.fit(X_train, Y_train)

best_lr = grid_fit.best_estimator_



best_cv_scores = cross_val_score(best_lr, X_train, Y_train, cv = 5)



best_lr.fit(X_train, Y_train)

predict = best_lr.predict(X_test)

score = best_lr.score(X_train, Y_train)

score2 = best_lr.score(X_test, Y_test)



print(score2)

print(predict)



submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predict})



filename = 'Titanic Predictions 3.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)


