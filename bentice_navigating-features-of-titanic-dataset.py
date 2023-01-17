#Load packages for manipulating data
import numpy as np
import pandas as pd

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings #not sure if I even need this
warnings.filterwarnings('ignore')
%matplotlib inline
train_data = pd.read_csv('../input/train.csv') # We will use the training data set for data exploration
test_data = pd.read_csv('../input/test.csv')
train_data.head()
train_data.info()
f,ax=plt.subplots(1,2,figsize=(18,8))
train_data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=train_data,ax=ax[1])
ax[1].set_title('Survived')
plt.show()
train_data[['Surname', 'Title']] = train_data.Name.str.extract('(.+?), (.+?)\.')
train_data[['Surname', 'Title']].head()
train_data['Title'].unique()
status_titles = ['Lady', 'Sir',
       'the Countess', 'Master', 'Don', 'Jonkheer', 'Dona'] #titles denoting status I took a peak at the test data to make sure I had all possible Titles.
earned_titles = ['Rev', 'Dr', 'Col', 'Major', 'Capt'] # titles earned
rare_titles = status_titles + earned_titles
correct_titles = {'Mlle' : 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}
print(train_data.loc[train_data['Title'].isin(rare_titles)][['Title', 'Survived']].mean())
train_data['Title'] = train_data['Title'].replace(correct_titles)
train_data['Title'] = train_data['Title'].replace(earned_titles, value= "Earned_Title")
train_data['Title'] = train_data['Title'].replace(status_titles, value= "Noble_Title")
train_data['Title'].unique()
print(train_data[['Title', 'Survived']].groupby(['Title']).mean())
sns.swarmplot(x='Survived', y='Age', hue = 'Title', data=train_data)
train_data.head()
train_data['Family_Members'] = train_data['SibSp'] + train_data['Parch']
men_data = train_data.loc[train_data['Sex'] == 'male']
print (train_data[['Family_Members','Sex',  'Survived']].groupby(['Family_Members'], as_index=False).mean())
f, ax = plt.subplots(figsize=(10, 10))
sns.countplot(x="Family_Members", hue='Survived', data=train_data, palette="Purples")
sns.countplot(x="Family_Members", hue='Survived', data=men_data, palette="Reds")

sns.factorplot(x="Family_Members", y="Survived", hue="Sex", data=train_data)
train_data['Family_Members'] = train_data['Family_Members'].replace([5, 6, 7, 8, 9, 10], value= 4)
men_data= train_data.loc[train_data['Sex'] == 'male']
f, ax = plt.subplots(figsize=(10, 10))
sns.countplot(x="Family_Members", hue='Survived', data=train_data, palette="Purples")
sns.countplot(x="Family_Members", hue='Survived', data=men_data, palette="Reds")
sns.factorplot(x="Family_Members", y="Survived", hue="Sex", data=train_data)
# put in a couple of KDE plots showing relations between Age, Deck, Port of Embarkation and Survival Rate
train_data[['Cabin_Letter', 'Cabin_Number']] = train_data.Cabin.str.extract('([A-Z])(\d*)') # Split the Variables up into floors and cabin numbers
train_data['Cabin_Letter'] = train_data['Cabin_Letter'].fillna('Unknown') # There might be a good reason we don't know what Cabin
train_data['Cabin_Number'] = train_data['Cabin_Number']
train_data[['Cabin_Letter', 'Cabin_Number']].head()
sns.factorplot(x="Cabin_Letter", kind="count", data=train_data )
sns.set(style="whitegrid", palette="muted")
sns.swarmplot(x='Cabin_Letter', y='Age', hue = 'Survived', data=train_data)
sns.swarmplot(x='Survived', y='Age', hue = 'Cabin_Letter', data=train_data)
mystery_age = train_data.loc[train_data['Age'].isnull()]
mys_men_age = mystery_age.loc[train_data['Sex'] == 'male']
mystery_age.Survived.mean(), train_data.Survived.mean()
mystery_age.Parch.mean(), mystery_age.Family_Members.mean(), mystery_age.SibSp.mean()
train_data.Parch.mean(), train_data.Family_Members.mean(), train_data.SibSp.mean()
sns.kdeplot(train_data.Family_Members)
sns.kdeplot(mystery_age.Family_Members)
sns.kdeplot(train_data.Age)
train_data.Age.describe()
train_data['CategoricalAge'] = pd.cut(train_data['Age'], [0, 17, 28, 38, 80])
print (train_data[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode())
print (train_data[['Embarked', 'Fare', 'Survived']].groupby(['Embarked'], as_index=False).mean())
train_data[['Embarked', 'Pclass', 'Fare']].groupby(['Embarked', 'Pclass']).describe()
train_data[['Fare']].describe()
C_em = train_data.loc[train_data['Embarked'] == 'C']
Q_em = train_data.loc[train_data['Embarked'] == 'Q']
S_em = train_data.loc[train_data['Embarked'] == 'S']
f, ax = plt.subplots(figsize=(10, 10))
sns.kdeplot(C_em.Fare, legend= True)
sns.kdeplot(Q_em.Fare, legend= True)
sns.kdeplot(S_em.Fare, legend= True)
sns.kdeplot(train_data.Fare, shade=True, legend= True)
ax.legend(labels = ['C', 'Q', 'S', 'Overall'])
ax.set_xlim(0, 300)
train_data['CategoricalFare'] = pd.qcut(train_data['Fare'], [0, .5, .95, 1])
print (train_data[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
train_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')
full_data = [train_set, test_set]
status_titles = ['Lady', 'Sir',
       'the Countess', 'Master', 'Don', 'Jonkheer', 'Dona'] #titles denoting status I took a peak at the test data to make sure I had all possible Titles.
earned_titles = ['Rev', 'Dr', 'Col', 'Major', 'Capt'] # titles earned
correct_titles = {'Mlle' : 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype('int64')
    
    # Mapping titles
    dataset[['Surname', 'Title']] = dataset.Name.str.extract('(.+?), (.+?)\.')
    dataset['Title'] = dataset['Title'].replace(correct_titles)
    dataset['Title'] = dataset['Title'].replace(earned_titles, value= "Earned_Title")
    dataset['Title'] = dataset['Title'].replace(status_titles, value= "Noble_Title")
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Earned_Title": 4, "Noble_Title": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Cabin_Letter
    dataset[['Cabin_Letter', 'Cabin_Number']] = dataset.Cabin.str.extract('([A-Z])(\d*)')
    dataset['Cabin_Letter'] = dataset['Cabin_Letter'].fillna('Unknown')
    dataset['Cabin_Letter'] = dataset['Cabin_Letter'].map({'Unknown': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T' : 8}).astype(int)
    
    #Mapping Family_Members
    dataset['Family_Members'] = dataset['SibSp'] + dataset['Parch']
    dataset['Family_Members'] = dataset['Family_Members'].replace([5, 6, 7, 8, 9, 10], value= 4)
    
    # Mapping Fare
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
    dataset.loc[dataset['Fare'] <= 14.454, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 112.079), 'Fare'] = 1
    dataset.loc[ dataset['Fare'] > 112.079, 'Fare'] 							        = 2
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 17, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 17) & (dataset['Age'] <= 28), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 28) & (dataset['Age'] <= 38), 'Age'] = 2
    dataset.loc[ dataset['Age'] > 38, 'Age']                           = 3
    dataset.loc[ dataset['Age'].isnull(), 'Age']                       = 4
    dataset['Age'] = dataset['Age'].astype(int)

# Feature Selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',
                 'Parch', 'Surname', 'Cabin_Number']
train_set = train_set.drop(drop_elements, axis = 1)
test_set  = test_set.drop(drop_elements, axis = 1)

train = train_set
test  = test_set

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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
	AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
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
candidate_classifier = GradientBoostingClassifier()
candidate_classifier.fit(train[0::, 1::], train[0::, 0])
result = candidate_classifier.predict(test)
PassengerId = test_data['PassengerId']
Submission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': result })
Submission.to_csv("submission.csv", index=False)