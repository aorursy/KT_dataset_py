%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import re as re



train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})

test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})

full_data = [train, test]
print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
print (train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())
for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
for dataset in full_data:

    age_avg 	   = dataset['Age'].mean()

    age_std 	   = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

train['CategoricalAge'] = pd.cut(train['Age'], 5)



print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
def get_title(name):

	title_search = re.search(' ([A-Za-z]+)\.', name)

	# If the title exists, extract and return it.

	if title_search:

		return title_search.group(1)

	return ""



for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)



print(pd.crosstab(train['Title'], train['Sex']))
for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
alpha = 0.6

fig = plt.figure(figsize=(8, 12))

grouped = train.groupby(['Survived'])

group0 = grouped.get_group(0)

group1 = grouped.get_group(1)



plot_rows = 5

plot_cols = 2

ax1 = plt.subplot2grid((plot_rows,plot_cols), (0,0), rowspan=1, colspan=1)

plt.hist([group0.Age, group1.Age], bins=16, range=(0,80), stacked=True, 

        label=['Not Survived', 'Survived'], alpha=alpha)

plt.legend(loc='best', fontsize='x-small')

ax1.set_title('Survival distribution by Age')



ax2 = plt.subplot2grid((plot_rows,plot_cols), (0,1), rowspan=1, colspan=1)

n, bins, patches = plt.hist([group0.Pclass, group1.Pclass], bins=5, range=(0,5), 

        stacked=True, label=['Not Survived', 'Survived'], alpha=alpha)

plt.legend(loc='best', fontsize='x-small')

ax2.set_xticks([1.5, 2.5, 3.5])

ax2.set_xticklabels(['Class1', 'Class2', 'Class3'], fontsize='small')

ax2.set_yticks([0, 150, 300, 450, 600, 750])

ax2.set_title('Survival distribution by Pclass')



ax3 = plt.subplot2grid((plot_rows,plot_cols), (1,0), rowspan=1, colspan=2)

ax3.set_title('Survival distribution by Sex')

patches, l_texts, p_texts = plt.pie(train.groupby(['Survived', 'Sex']).size(), 

        labels=['Not Survived Female', 'Not Survived Male', 'Survived Female', 'Survived Male'],

        autopct='%3.1f', labeldistance = 1.1, pctdistance = 0.6)

plt.legend(loc='upper right', fontsize='x-small')

for t in l_texts:

    t.set_size(10)

for p in p_texts:

    p.set_size(10)

#plt.legend(loc='best', fontsize='x-small')

plt.axis('equal')



plt.subplots_adjust(wspace=0.3, hspace=0.3)
for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4



# Feature Selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',\

                 'Parch', 'FamilySize']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)



test  = test.drop(drop_elements, axis = 1)



print (train.head(10))



train = train.values

test  = test.values
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression



classifiers = [DecisionTreeClassifier(),

    RandomForestClassifier(),

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
candidate_classifier =  RandomForestClassifier()

candidate_classifier.fit(train[0::, 1::], train[0::, 0])

result = candidate_classifier.predict(test)

idarr = []

for i in range(892,1310):

    idarr.append(i)

submission = pd.DataFrame({"ID": idarr, "Survived": result })

submission.to_csv('titanic.csv', index=False)
