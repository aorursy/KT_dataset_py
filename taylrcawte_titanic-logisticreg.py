# standard libs

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# sklearn libs

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import make_scorer, confusion_matrix

from sklearn.model_selection import learning_curve



sns.set_style('darkgrid')



# The following line is needed to show plots inline in notebooks

%matplotlib inline 
train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')
train_data.head()
train_data.describe()
test_data.head()
test_data.describe()
sns.countplot(x='Survived', hue='Sex', data=train_data)
sns.countplot(x='Survived', hue='Pclass', data=train_data)
plt.hist(x='Fare', data=train_data, bins=10)
fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(train_data.isnull(), cmap='plasma', cbar=False, yticklabels=False, ax=ax)

plt.title('NaN values in the training dataset')
fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(test_data.isnull(), cmap='plasma', cbar=False, yticklabels=False, ax=ax)

plt.title('NaN values in the test dataset')
train_data[train_data.Embarked.isnull()]

test_data[test_data.Fare.isnull()]
train_data.dropna(axis=0, subset=['Embarked'], inplace=True)



# verify the data has been dropped

train_data[train_data.Embarked.isnull()]
test_data.dropna(axis=0, subset=['Fare'], inplace=True)



# verify the data has been dropped

test_data[test_data.Fare.isnull()]
train_data[train_data.Age.isnull()]
fig, ax = plt.subplots()

sns.boxplot(x='Pclass', y='Age', data=train_data)
train_data.groupby('Pclass')['Age'].mean()
def fill_age(age, pclass):

    if age == age: 

        return age

    if pclass == 1: 

        return 38

    elif pclass == 2: 

        return 30

    else: 

        return 25

    

train_data['Age'] = train_data.apply(lambda x: fill_age(x['Age'], x['Pclass']), axis=1)

test_data['Age'] = test_data.apply(lambda x: fill_age(x['Age'], x['Pclass']), axis=1)
train_data[train_data.Age.isnull()]
test_data[test_data.Age.isnull()]
kde = sns.FacetGrid(train_data, hue='Survived', aspect=4)

kde.map(sns.kdeplot, 'Age', shade=True)

kde.add_legend()
train_data.drop('Cabin', axis=1, inplace=True)

test_data.drop('Cabin', axis=1, inplace=True)
# verify no missing data left 

fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(test_data.isnull(), cmap='plasma', cbar=False, yticklabels=False, ax=ax)

plt.title('NaN values in the test dataset')
fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(test_data.isnull(), cmap='plasma', cbar=False, yticklabels=False, ax=ax)

plt.title('NaN values in the test dataset')
train_data['Family'] = train_data['SibSp'] + train_data['Parch']

test_data['Family'] = test_data['SibSp'] + test_data['Parch']



train_data.head()
train_data.drop(['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch'], axis=1, inplace=True)

test_data.drop(['Name', 'Ticket', 'SibSp', 'Parch'], axis=1, inplace=True)



train_data.head()
dummy_sex = pd.get_dummies(train_data['Sex'])

dummy_embarked = pd.get_dummies(train_data['Embarked'])



# add the dummy variables to the the training dataframe 

train_data = pd.concat([train_data, dummy_sex, dummy_embarked], axis=1)



# drop 1 of each of the dummies as it is implied by other dummy columns

# remaining dummy variables are 'female', 'C', and 'Q'

# also drop the original data columns prior to being converted to dummy variables

train_data.drop(['Sex', 'Embarked', 'male', 'S'], inplace=True, axis=1)

train_data.head()
dummy_sex = pd.get_dummies(test_data['Sex'])

dummy_embarked = pd.get_dummies(test_data['Embarked'])



# add the dummy variables to the the training dataframe 

test_data = pd.concat([test_data, dummy_sex, dummy_embarked], axis=1)



# drop 1 of each of the dummies as it is implied by other dummy columns

# remaining dummy variables are 'female', 'C', and 'Q'

# also drop the original data columns prior to being converted to dummy variables

test_data.drop(['Sex', 'Embarked', 'male', 'S'], inplace=True, axis=1)

test_data.head()
X = train_data.drop(['Survived'],axis=1)

y = train_data['Survived']
model = LogisticRegression()

scaler = StandardScaler()

kfold = KFold(n_splits=10)

kfold.get_n_splits(X)



best_model = model

best_params = {}

best_accuracy = 0

best_std = 0



for C in [0.001,0.01,0.05,0.1,0.5,1,5,10, 100]:

    for solver in ['newton-cg','lbfgs','liblinear','sag']:

        

        model = LogisticRegression(C=C, solver=solver)

        accuracy = np.zeros(10)

        np_idx = 0

        

        for train_idx, test_idx in kfold.split(X):

            X_train, X_test = X.values[train_idx], X.values[test_idx]

            y_train, y_test = y.values[train_idx], y.values[test_idx]



            X_train = scaler.fit_transform(X_train)

            X_test = scaler.transform(X_test)



            model.fit(X_train, y_train)



            predictions = model.predict(X_test)



            TN = confusion_matrix(y_test, predictions)[0][0]

            FP = confusion_matrix(y_test, predictions)[0][1]

            FN = confusion_matrix(y_test, predictions)[1][0]

            TP = confusion_matrix(y_test, predictions)[1][1]

            total = TN + FP + FN + TP

            ACC = (TP + TN) / float(total)



            accuracy[np_idx] = ACC*100

            np_idx += 1

        

        if np.mean(accuracy) > best_accuracy:

            best_model = model

            best_params = {'C':C, 'solver':solver}

            best_accuracy = np.mean(accuracy)

            best_std = np.std(accuracy)



print (best_params)

print ("Best Score: {}%({}%)".format(round(best_accuracy,3),round(best_std,3)))      



print ("\nThe optimal log model uses C={}, and a {} solver, and has a cross validation score of {}% with a standard deviation of {}%".format(best_params['C'],best_params['solver'],round(best_accuracy,3),round(best_std,3)))
model = LogisticRegression(C=best_params['C'],solver=best_params['solver'])

scaler = StandardScaler()



X_train = train_data.drop(['Survived'],axis=1)

y_train = train_data['Survived']



X_test = test_data.drop('PassengerId', axis=1)



X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



model.fit(X_train, y_train)



predictions = model.predict(X_test)
new_preds=[]



for i in range(len(predictions)):

    if predictions[i] == 0: 

        new_preds.append('deceased')

    else: 

        new_preds.append('survived')
labels, counts = np.unique(new_preds, return_counts=True)

plt.bar(labels, counts, align='center')

plt.gca().set_xticks(labels)

plt.title('Predicted Survival Rate of Passengers in the Titanic Test Set')

plt.show()
submission_df = pd.DataFrame({'PassengerId': test_data.PassengerId.astype(int), 'Survived': predictions})
# save the submission file to the working output directory 

submission_df.to_csv('./titanic_submission.csv', index=False)