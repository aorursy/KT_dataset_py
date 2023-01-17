import numpy as np 

import pandas as pd 



import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



from sklearn import linear_model

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import Perceptron

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier
data_root = 'drive/TitanicData/titanic/'

train_df = pd.read_csv('{}{}'.format(data_root,'train.csv'))

test_df = pd.read_csv('{}{}'.format(data_root,'test.csv'))
train_df.head(10)
train_df.info()
train_df.describe()
train_df.shape
print("Null: ", train_df.Age.isnull().sum(), "  Missing Data in percentage: ",train_df.Age.isnull().sum()/train_df.shape[0]*100)

print("Null: ", train_df.Sex.isnull().sum(), "  Missing Data in percentage: ",train_df.Sex.isnull().sum()/train_df.shape[0]*100)

print("Null: ", train_df.Cabin.isnull().sum(), "  Missing Data in percentage: ",train_df.Cabin.isnull().sum()/train_df.shape[0]*100)

print("Null: ", train_df.Fare.isnull().sum(), "  Missing Data in percentage: ",train_df.Fare.isnull().sum()/train_df.shape[0]*100)

print("Null: ", train_df.Pclass.isnull().sum(), "  Missing Data in percentage: ",train_df.Pclass.isnull().sum()/train_df.shape[0]*100)

print("Null: ", train_df.Embarked.isnull().sum(), "  Missing Data in percentage: ",train_df.Embarked.isnull().sum()/train_df.shape[0]*100)

print("Null: ", train_df.Survived.isnull().sum(), "  Missing Data in percentage: ",train_df.Survived.isnull().sum()/train_df.shape[0]*100)
men = train_df[train_df['Sex'] == 'male']

women = train_df[train_df['Sex'] == 'female']



print("Percentage of men survived: ", men.Survived.sum()/men.shape[0]*100)

print("Percentage of women survived: ", women.Survived.sum()/women.shape[0]*100)
sns.barplot(x='Sex', y='Survived', data=train_df)
sns.barplot(x='Age', y='Survived', data=women)


sns.barplot(x='Age', y='Survived', data=men)
sns.barplot(x='Pclass', y='Survived', data=train_df)
new_data = train_df[['Survived','SibSp','Parch']]

relatives = train_df['SibSp']+train_df['Parch']



new_data['Relatives'] = relatives
new_data.info()
new_data
sns.barplot(x='Relatives', y='Survived', data=new_data)
axes = sns.factorplot('Relatives','Survived', 

                      data=new_data, aspect = 4, )
sns.barplot(x='Embarked', y='Survived', data=train_df)
survived = train_df[train_df['Survived'] == 1]['Embarked'].value_counts()

dead = train_df[train_df['Survived'] == 0]['Embarked'].value_counts()

df = pd.DataFrame([survived,dead])

df.index = ['Survived','Dead']

df.plot(kind='bar', stacked=False, figsize=(15,5))
dead = train_df[train_df['Survived'] == 0]['Fare'].mean()

alive = train_df[train_df['Survived'] == 1]['Fare'].mean()



x_axis = ["dead","alive"]

y_axis = [dead,alive]



plt.bar(x_axis, y_axis, 0.5, color='black')

plt.xlabel('Dead or Alive')

plt.ylabel('Mean Fare')

plt.show()
mean_age_train = train_df['Age'].mean()

mean_age_test = test_df['Age'].mean()

std_age_train = train_df['Age'].std() 

std_age_test = test_df['Age'].std()



lt = []

for data in train_df['Age']:

  if np.isnan(data):

    x = np.random.randint(mean_age_train-std_age_train, mean_age_train+std_age_train)

    lt.append(x)

  else:

    lt.append(data)

    

train_df['Age'] = lt



lt = []

for data in test_df['Age']:

  if np.isnan(data):

    x = np.random.randint(mean_age_test-std_age_test, mean_age_test+std_age_test)

    lt.append(x)

  else:

    lt.append(data)

    

test_df['Age'] = lt
fare_mean = test_df['Fare'].mean()



lt = []

for data in test_df['Fare']:

  if np.isnan(data):

    x = fare_mean

    lt.append(x)

  else:

    lt.append(data)

    

test_df['Fare'] = lt
val = 'S'

train_df['Embarked'] = train_df['Embarked'].fillna(val)
print(test_df['Fare'].isnull().sum())

print(train_df['Fare'].isnull().sum())



print(test_df['Age'].isnull().sum())

print(train_df['Age'].isnull().sum())



print(test_df['Pclass'].isnull().sum())

print(train_df['Pclass'].isnull().sum())



print(test_df['Embarked'].isnull().sum())

print(train_df['Embarked'].isnull().sum())



print(test_df['Sex'].isnull().sum())

print(train_df['Sex'].isnull().sum())
def wrangle(data):

  

    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}).astype(int)

    

    new_embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')

    data = pd.concat([data, new_embarked], axis=1)

    return data.drop('Embarked', axis=1)

  

train_df = wrangle(train_df)

test_df = wrangle(test_df)
print(train_df.head())

print(test_df.head())
print(train_df.info())

print(test_df.info())
X_train = train_df.drop("Survived", axis=1)

X_train = X_train.drop("PassengerId", axis=1)

X_train = X_train.drop("SibSp", axis=1)

X_train = X_train.drop("Parch", axis=1)

X_train = X_train.drop("Ticket", axis=1)

X_train = X_train.drop("Cabin", axis=1)

X_train = X_train.drop("Name", axis=1)

Y_train = train_df["Survived"]

X_test = test_df

X_test = X_test.drop("PassengerId", axis=1)

X_test = X_test.drop("SibSp", axis=1)

X_test = X_test.drop("Parch", axis=1)

X_test = X_test.drop("Ticket", axis=1)

X_test = X_test.drop("Cabin", axis=1)

X_test = X_test.drop("Name", axis=1)
X_test.info()

X_train.info()
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

train_accuracy = round(random_forest.score(X_train, Y_train) * 100, 2)

Y_predR = random_forest.predict(X_test)

print(train_accuracy)
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, Y_train)

train_accuracy = round(knn.score(X_train, Y_train) * 100, 2)

Y_predK = knn.predict(X_test)

print(train_accuracy)
svc = SVC()

svc.fit(X_train, Y_train)

train_accuracy = round(svc.score(X_train, Y_train) * 100, 2)

Y_predS = svc.predict(X_test)

print(train_accuracy)
model_logistic = LogisticRegression()

model_logistic.fit(X_train, Y_train)

train_accuracy = round(model_logistic.score(X_train, Y_train) * 100, 2)

Y_predL = model_logistic.predict(X_test)

print(train_accuracy)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_predR

    })

submission.to_csv('{}{}'.format(data_root,'submission.csv'), index=False)