

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train_data = pd.read_csv('/kaggle/input/titanic/train.csv')



train_data.head()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.head()
missing_data = train_data.isnull()

for column in missing_data.columns.values.tolist():

    print(column)

    print(missing_data[column].value_counts())

    print('')
medianage = train_data['Age'].median(skipna=True)

meanage = train_data['Age'].mean(skipna=True)

print('the mean of the age is :', meanage, 'the median of the age is:', medianage)
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)

train_data.head(10)
for i, col in enumerate(['SibSp', 'Parch']):

    plt.figure(i)

    sns.catplot(x=col, y='Survived', data=train_data, kind='point', aspect=2, )
train_data['Family count'] = train_data['SibSp'] + train_data['Parch']
train_data.drop(['SibSp', 'Parch'], axis=1, inplace=True)
train_data.isnull().sum()


train_data.groupby(train_data['Cabin'].isnull())['Survived'].mean()
train_data['Cabin_ind'] = np.where(train_data['Cabin'].isnull(),0,1)

train_data.head()
gender_num = {'male':0 , 'female':1}

train_data['Sex']= train_data['Sex'].map(gender_num)


train_data.drop(['Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

train_data.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics



from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier

features = train_data.drop('Survived',axis=1)

labels = train_data['Survived']



X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size= 0.4)

def print_results(results):

    print('BEST PARAMS: {}\n'.format(results.best_params_))

    

    means=results.cv_results_['mean_test_score']

    stds = results.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, results.cv_results_['params']):

        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
lr = LogisticRegression()

parameters = {

    'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]

}

cv = GridSearchCV(lr, parameters, cv=5)

cv.fit(X_train, y_train.values.ravel())



print_results(cv)

           
cv.best_estimator_
svc = SVC()

parameters= {

    'kernel': ['linear', 'rbf'],

    'C': [0.1, 1, 10]

}

cv = GridSearchCV(svc, parameters, cv=5)

cv.fit(X_train, y_train.values.ravel())



print_results(cv)
cv.best_estimator_
mlp = MLPClassifier()

parameters= {

   'hidden_layer_sizes': [(10,), (50,), (100,)],

    'activation': ['relu', 'tanh', 'logistic'],

    'learning_rate': ['constant', 'invscaling', 'adaptive']

}

cv = GridSearchCV(mlp, parameters, cv=5)

cv.fit(X_train, y_train.values.ravel())



print_results(cv)
cv.best_estimator_
rf = RandomForestClassifier()

parameters= {

    'n_estimators':[5, 50, 250],

    'max_depth':[2, 4, 8, 16, 32, None]

    

}

cv=GridSearchCV(rf, parameters, cv=5)

cv.fit(X_train, y_train.values.ravel())



print_results(cv)
cv.best_estimator_
gb = GradientBoostingClassifier()

parameters= {

    'n_estimators': [5, 50, 250, 500],

    'max_depth': [1, 3, 5, 7, 9],

    'learning_rate': [0.01, 0.1, 1, 10, 100]

}

cv = GridSearchCV(gb, parameters, cv=5)

cv.fit(X_train, y_train.values.ravel())



print_results(cv)
cv.best_estimator_
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)



test_data['Family count'] = test_data['SibSp'] + test_data['Parch']



test_data['Cabin_ind'] = np.where(test_data['Cabin'].isnull(),0,1)



gender_num = {'male':0 , 'female':1}

test_data['Sex']= test_data['Sex'].map(gender_num)



test_data.drop(['Cabin', 'Embarked', 'Name', 'Ticket', 'SibSp', 'Parch'], axis=1, inplace=True)

test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)

test_data.head()
GB = GradientBoostingClassifier(n_estimators=50, max_depth = 1, learning_rate=1)

GB.fit(X_train, y_train.values.ravel())

predictions = GB.predict(test_data)
output1 = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output1.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")