%matplotlib inline

import numpy as np

import pandas as pd

import re as re



train = pd.read_csv('../input/titanic/train.csv', header = 0, dtype={'Age': np.float64})

test  = pd.read_csv('../input/titanic/test.csv' , header = 0, dtype={'Age': np.float64})

print(type(train))

full_data = [train, test]



print (train.info())
print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
print (train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())
for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[ dataset['SibSp'] + dataset['Parch'] + 1 == 1, 'IsAlone'] = 1

print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
for dataset in full_data:

    age_avg 	   = dataset['Age'].mean()

    age_std 	   = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    
for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 2, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] >  2) & (dataset['Age'] <=  8), 'Age'] = 1

    dataset.loc[(dataset['Age'] >  8) & (dataset['Age'] <= 16), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 5

    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 6



# Feature Selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'SibSp',\

                 'Parch', 'Embarked']

train = train.drop(drop_elements, axis = 1)

test  = test.drop(drop_elements, axis = 1)



print (train.head(10))



train = train.values

test  = test.values



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression



classifiers = [

    KNeighborsClassifier(5),

    SVC(probability=True),

    DecisionTreeClassifier(),

    LogisticRegression()]



log_cols = ["Classifier", "Accuracy"]

log 	 = pd.DataFrame(columns=log_cols)



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)



X = train[0::, 1::]

y = train[0::, 0]



acc_dict = {}



for train_index, test_index in sss.split(X, y):

	X_train, X_test = X[train_index], X[test_index]

	y_train, y_test = y[train_index], y[test_index]

	

	for clf in classifiers:

		name = clf.__class__.__name__

		clf.fit(X_train, y_train)

		train_predictions = clf.predict(X_test)

		acc = accuracy_score(y_test, train_predictions)

		if name in acc_dict:

			acc_dict[name] += acc

		else:

			acc_dict[name] = acc



for clf in acc_dict:

	acc_dict[clf] = acc_dict[clf] / 10.0

	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

	log = log.append(log_entry)



plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
candidate_classifier = SVC()

candidate_classifier.fit(train[0::, 1::], train[0::, 0])

result = candidate_classifier.predict(test)
