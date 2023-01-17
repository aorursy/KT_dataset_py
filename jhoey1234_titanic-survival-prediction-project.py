# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid")



from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

#from empiricaldist import Pmf, Cdf



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def LogRegModel(X, y):

    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y) 

    logreg = LogisticRegression()

    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    score = logreg.score(X_test, y_test)

    matrix = confusion_matrix(y_test, y_pred)

    #print('The score of the model is {}'.format(score))

    #print('The confusion matrix is: ', matrix)

    return score, matrix
def KNeighbors(X, y):

    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    k = KNeighborsClassifier(n_neighbors=3)

    k.fit(X_train, y_train)

    y_pred = k.predict(X_test)

    score = k.score(X_test, y_test)

    matrix = confusion_matrix(y_test, y_pred)

    #print('The score of the model is {}'.format(score))

    #print('The confusion matrix is: ', matrix)

    return score, matrix
def TreeClass(X, y):

    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    tree = DecisionTreeClassifier()

    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)

    score = tree.score(X_test, y_test)

    matrix = confusion_matrix(y_test, y_pred)

    #print('The score of the model is {}'.format(score))

    #print('The confusion matrix is: ', matrix)

    return score, matrix
def ecdf(dataseries):

    ### Compute Empirical CDF for 1D dataseries ###

    n = len(dataseries)

    x = np.sort(dataseries)

    y = np.arange(1, n+1)/n

    return x, y
#read in train and test data sets

train = pd.read_csv('/kaggle/input/titanic/train.csv') 

test = pd.read_csv('/kaggle/input/titanic/test.csv')
#check shape of  datasets

print('train: ',train.shape)

print('test: ',test.shape)
display(train.head())
print(train.info())

print(test.info())
data_list = [train, test] #create list of both datasets to easily replicate manipulation



#drop passenger id, name, cabin,and ticket

for dataset in data_list:

    dataset.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

    

#Handle missing data in both datasets

for dataset in data_list:

    dataset['Age'].fillna(dataset.Age.mean(), inplace=True)

    dataset['Embarked'].fillna(dataset.Embarked.mode()[0], inplace=True)

    

#fill in missing value for missing fare in test set     

test['Fare'].fillna(test['Fare'].median(), inplace=True)
#check discrete object columns for incorrect or redundanct values

#print(train.Sex.dtype)

display(train['Sex'].unique())

#print(train.Embarked.dtype)

display(train['Embarked'].unique())
#convert object classes to category classes and then convert to numerical features

for dataset in data_list:

    for column in dataset:

        if dataset[column].dtype == 'object':

            dataset[column] = dataset[column].astype('category')

            dataset[column] = LabelEncoder().fit_transform(dataset[column])

            

#check to see that there are no missing

print(train.info())

print(test.info())
train.head()
#initialize random seed

np.random.seed(0)
#create list for training features and labels

features = list(train.columns.drop('Survived'))

X = train[features]

y = train['Survived']
model1 = LogRegModel(X, y)
print('Number of Survivors: {}'.format(train.Survived.sum())) #total number of survivors in training set

print('Number of Deaths: {}'.format(train[train['Survived'] == 0]['Survived'].count())) #total number of deaths in training set
print('Total number of males onboard: {}'.format(train[train['Sex'] == 0]['Sex'].count()))

print('Total number of females onboard: {}'.format(train[train['Sex'] == 1]['Sex'].count()))

sns.countplot(data=train, x='Survived', hue='Sex')

plt.title('Number of Survivors Male vs. Female')

plt.show()
age_dist = sns.FacetGrid(train, col='Survived')

age_dist.map(sns.distplot, 'Age')

plt.show()



sns.distplot(train['Age'])

plt.ylabel

plt.show()
for dataset in data_list:

    dataset['Age'] = pd.cut(dataset['Age'], bins=[0,4,9,14,19,24,35,50,65,120], labels=['Toddler', 'Young Child', 'Child', 'Teenage', 'Young Adult', 'Adult','Older Adult', 'Middle Age', 'Senior'])
Age_Group_Count = train.groupby('Age')['Survived'].sum()

Age_Group_Proportion = train.groupby('Age')['Survived'].mean()

Age_Group_Count

Age_Group_Proportion
print(train['Age'].value_counts())

fig, ax = plt.subplots(1, 2, figsize=(10, 10))



#fig.subplots_adjust(wspace=1)



Age_Group_Count.plot(kind='bar', ax=ax[0])

ax[0].set_title('Total Survived per Age Group')



Age_Group_Proportion.plot(kind='bar', ax=ax[1])

ax[1].set_title('Percent survived per Age Group')



plt.show()
for dataset in data_list:

    dataset['Age'] = LabelEncoder().fit_transform(dataset['Age'])
#see distribution of survived based of embarked location

fig, ax = plt.subplots(1, 3, figsize=(15,10))

fig.subplots_adjust(wspace=0.5)

sns.countplot(data=train, x='Survived', hue='Embarked', ax=ax[0])

ax[0].set_title('Suvival and Death numbers per Embarked')



sns.countplot(data=train, x='Embarked', ax=ax[1])

ax[1].set_title('Total Passengers Embarked from each location')



ax[2] = embarked_mean = train.groupby('Embarked')['Survived'].mean().plot(kind='bar', ax=ax[2])

ax[2].set_title('Survival % from each location')

plt.show()





for dataset in data_list:

    dataset.drop('Embarked', axis=1, inplace=True)
#see distribution of survived based of embarked location

fig, ax = plt.subplots(1, 3, figsize=(15,10))

fig.subplots_adjust(wspace=0.5)

sns.countplot(data=train, x='Survived', hue='Pclass', ax=ax[0])

ax[0].set_title('Suvival and Death numbers per Economic Class')



sns.countplot(data=train, x='Pclass', ax=ax[1])

ax[1].set_title('Total Passengers Embarked from each Economic Class')



ax[2] = embarked_mean = train.groupby('Pclass')['Survived'].mean().plot(kind='bar', ax=ax[2])

ax[2].set_title('Survival % from each economic class')

plt.show()
print(train['SibSp'].value_counts())

print(train['Parch'].value_counts())
fig, ax = plt.subplots(1,2, figsize = (8, 10))

sns.countplot(data=train, x='Parch', hue='Survived', ax=ax[0])

sns.countplot(data=train, x='SibSp', hue='Survived', ax=ax[1])

plt.show()
# for dataset in data_list:

# #     dataset['Total Family'] = pd.cut(dataset['Total Family'], bins=[-1,3,6,16], labels=['Small', 'Medium', 'Large'])

# #     dataset['Total Family'] = LabelEncoder().fit_transform(dataset['Total Family'])

#     dataset.drop(['Parch', 'SibSp'], axis=1, inplace=True)
fare_x_ecdf, fare_y_ecdf = ecdf(train['Fare'])

sns.lineplot(x=fare_x_ecdf, y=fare_y_ecdf)

plt.xlabel('Ticket Fare')

plt.ylabel('Pecentile of Ticket Fare')

plt.title('ECDF of Ticket Fare ')

plt.show()
#create groups for price paid for ticket

percentiles = [20, 40, 60, 80]

print(train['Fare'].min())

print(np.percentile(train['Fare'], percentiles))

print(train['Fare'].max())
fare_bins = np.percentile(train['Fare'], percentiles)

fare_bins = np.append(fare_bins, np.max(train['Fare'])+1)

fare_bins = np.insert(fare_bins, 4, 100)

fare_bins = np.insert(fare_bins, 0, np.min(train['Fare'])-1)

#print(fare_bins)

for dataset in data_list:

    dataset['Fare'] = pd.cut(dataset['Fare'], bins=fare_bins, labels=['VeryLow', 'Low', 'LowMed', 'Medium','MedHigh', 'High'])

sns.countplot(x='Fare', hue='Survived', data=train)

plt.show()
for dataset in data_list:

    dataset['Fare'] = LabelEncoder().fit_transform(dataset['Fare'])

    
features = list(train.columns.drop(['PassengerId','Survived']))

X = train[features]

y = train['Survived']

X_test_data = test[features]
log_model2 = LogRegModel(X, y)

k_model = KNeighbors(X, y)

tree_model = TreeClass(X, y)
print(model1)

print(log_model2)

print(tree_model)

print(k_model)
tree = DecisionTreeClassifier()

tree.fit(X, y)

y_pred_test = tree.predict(X_test_data)
test_id = test['PassengerId']

test_results = y_pred_test
results = pd.DataFrame({'PassengerId': test_id, 'Survived': test_results})

results.to_csv('survival_predictions.csv', index=False)