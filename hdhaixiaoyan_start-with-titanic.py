# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import seaborn as sns

import matplotlib.pyplot as plt



import re
# load the data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

data = [train, test]



PassengerId = test['PassengerId']
train.head(3)
test.head(3)
train.info()

print('-'*100)

test.info()
train.describe()
train.describe(include=['O'])
#Pcalsss

#There is no missing value on this feature and it is a categorical value. 

#I want to see it's impact on our train set

pclass_Sur = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()

print (pclass_Sur)

sns.barplot(x=pclass_Sur.index, y=pclass_Sur['Survived'])
#Names are unique across the dataset

#I want to extract two features from name:the length of the name and title

#the length of name

train['Name_length'] = train['Name'].apply(len)

test['Name_length'] = test['Name'].apply(len)

namelen_Sur = train[['Name_length', 'Survived']].groupby(['Name_length'], as_index=False).mean()

sns.barplot(x=namelen_Sur.Name_length, y=namelen_Sur['Survived'])

#title

def get_title(name):

	title_search = re.search(' ([A-Za-z]+)\.', name)

	if title_search:

		return title_search.group(1)

	return ""



for dataset in data:

    dataset['Title'] = dataset['Name'].apply(get_title)



print(pd.crosstab(train['Title'], train['Sex']))
for dataset in data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



name_Sur = train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

sns.barplot(x=name_Sur.Title, y=name_Sur['Survived'])
#sex has two possible values 

sex_Sur = train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()

sns.barplot(x=sex_Sur.Sex, y=sex_Sur['Survived'])
#age

#There are some missing values exist in columns Age

#I choose the random numbers between (mean - std) and (mean + std) to fill the missing values.

for dataset in data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)



# then let us look the distribute of the age

facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()
#as there so many values in age ,we can cut the age

bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeCat'] = pd.cut(train['Age'], bins, labels=group_names)

print (train.head())

# There are two columns Parch & SibSp 

# we can have only one column represent if the passenger had any family member aboard or not,

for dataset in data:

    dataset['Family'] = dataset['Parch'] + dataset['SibSp']

    dataset['Family'].loc[dataset['Family'] > 0] = 1

    dataset['Family'].loc[dataset['Family'] == 0] = 0

# drop Parch & SibSp

train = train.drop(['SibSp','Parch'], axis=1)

test = test.drop(['SibSp','Parch'], axis=1)



# plot

fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))



# sns.factorplot('Family',data=titanic_df,kind='count',ax=axis1)

sns.countplot(x='Family', data=train, order=[1,0], ax=axis1)



# average of survived for those who had/didn't have any family member

family_perc = train[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()

sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)



axis1.set_xticklabels(["With Family","Alone"], rotation=0)
#There are some missing value in Fare,we can fill them with median

train['Fare'] = train['Fare'].fillna(train['Fare'].median())

test['Fare'] = test['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
#there are too many values missed in Cabin,so we drop this feature

#Embarked feature takes S, Q, C values based on port of embarkation. 

#Our training dataset has two missing values. We simply fill these with the most common occurance.

freq_port = train.Embarked.dropna().mode()[0]

train['Embarked'] = train['Embarked'].fillna(freq_port)  

train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for dataset in train,test:

    dataset['Sex'] = dataset['Sex'].map({'female':1,'male':0}).astype(int)

   

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping).astype(int)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Agebins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    dataset.loc[ dataset['Age'] <= 0, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 0) & (dataset['Age'] <= 5), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 5) & (dataset['Age'] <= 12), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 12) & (dataset['Age'] <= 18), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 25), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 35), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 60), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 60, 'Age']                           = 7



train.head()
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['AgeCat', 'CategoricalFare'], axis = 1)



test  = test.drop(drop_elements, axis = 1)



print (train.head(10))
plt.figure(figsize=(10,10))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, linecolor='white', annot=True)
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

X = train.drop('Survived',axis=1)

y = train.Survived



clf = RandomForestClassifier()

parameters = {'n_estimators': [100, 300], 'max_features': [3, 4, 5, 'auto'],

              'min_samples_leaf': [9, 10, 12], 'random_state': [7]}

grid_search = GridSearchCV(estimator=clf, param_grid=parameters, cv=10, scoring='accuracy')

print("parameters:")

grid_search.fit(X, y)

print("Best score: %0.3f" % grid_search.best_score_)

print("Best parameters set:")

bsp = grid_search.best_estimator_.get_params()  # the dict of parameters with best score

for param_name in sorted(bsp.keys()):

    print("\t%s: %r" % (param_name, bsp[param_name]))

bsp
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

parameters = {'n_estimators': 100, 'max_features': 4, 'min_samples_leaf': 9, 'random_state': 7}

rf = RandomForestClassifier(**parameters)

rf.fit(X_train,y_train)



y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print (acc)



features = X_train.columns

importances = rf.feature_importances_



feature_important = pd.DataFrame(index=features, data=importances, columns=['importance'])

feature_important = feature_important.sort_values(by=['importance'], ascending=True)

feature_important.plot(kind='barh', stacked=True,figsize=(8, 5))
result = rf.predict(test)

Submission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': result })

Submission.to_csv("Submission.csv", index=False)