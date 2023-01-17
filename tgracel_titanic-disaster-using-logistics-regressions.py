# data Analysis

import numpy as np

import pandas as pd



# Data Visualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



#Machine Learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC
data1=pd.read_csv('../input/train.csv')   #read training set

data2=pd.read_csv('../input/test.csv')    #Read test set

combine=[data1,data2]
data1.head()
data1["Survived"].value_counts()
data1["Sex"].value_counts()
senior=0

adults=0

children=0

infant=0

for val in data1["Age"]:

    if val>=64:

        senior+=1

    else:

        if val>17:

            adults+=1

        else:

            if val>3:

                children+=1

            else:

                infant+=1

print("Number of Infants:",infant)

print("Number of Children:",children)

print("Number of Adults:",adults)

print("Number of Senior Citizens:",senior)
fam=0

for val in data1["SibSp"]:

    if val>0.0:

        fam+=1

for val in data1["Parch"]:

    if val>0.0:

        fam+=1

print("Number of passengers boarding with family or sibling: ", fam)
#CHECKING OUT FOR MISSING VALUES

for col in data1:

    val=data1[col].isnull().sum()

    if val>0.0:

        print("Number of missing values in column ",col,":",val)

data1.describe()
freq_1=pd.crosstab(index=data1['Survived'],columns=data1['Pclass'])

freq_2=pd.crosstab(index=data1['Survived'],columns=data1['Sex'])

freq_3=pd.crosstab(index=data1['Survived'],columns=data1['SibSp'])

freq_4=pd.crosstab(index=data1['Survived'],columns=data1['Parch'])

freq_5=pd.crosstab(index=data1['Survived'],columns=data1['Pclass'])

freq_6=pd.crosstab(index=data1['Survived'],columns=data1['Embarked'])

l=[freq_1,freq_2,freq_3,freq_4,freq_5,freq_6]

for i in l:

    print(i)

    print('_'*40)
#correlation matrix to know important features in prediction

corr=data1.corr()

sns.heatmap(corr[['Age','Fare']],annot=True,linewidth=0.1)

plt.show()
#correlation matrix to know important features in prediction

corr=data1.corr()

sns.heatmap(corr[['Survived','SibSp']],annot=True,linewidth=0.1)

plt.show()
#correlation matrix to know important features in prediction

corr=data1.corr()

sns.heatmap(corr[['Age','Survived']],annot=True,linewidth=0.1)

plt.show()
#correlation matrix to know important features in prediction

corr=data1.corr()

sns.heatmap(corr[['Survived','Parch']],annot=True,linewidth=0.1)

plt.show()
#correlation matrix to know important features in prediction

corr=data1.corr()

sns.heatmap(corr[['Fare','Parch','SibSp','Age','Pclass','Survived']],annot=True,linewidth=0.1)

plt.show()
# grid = sns.FacetGrid(data1, col='Pclass', hue='Survived')

grid = sns.FacetGrid(data1, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
print("Before", data1.shape, data2.shape, combine[0].shape, combine[1].shape)



data1 = data1.drop(['Ticket','Cabin'], axis=1)

data2 = data2.drop(['Ticket','Cabin'], axis=1)

combine = [data1, data2]



"After", data1.shape, data2.shape, combine[0].shape, combine[1].shape
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(data1['Title'], data1['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

data1[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



data1.head()
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



data1.head()
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')

grid = sns.FacetGrid(data1, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
guess_ages = np.zeros((2,3))

guess_ages
for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)

data1
freq_port = data1.Embarked.dropna().mode()[0]

freq_port

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

data1[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

data1.head()
data1 = data1.drop(['Embarked'], axis=1)

combine = [data1, data2]

data1.head()
X_train = data1[["Age","Parch","Sex"]]

Y_train = data1["Survived"]

X_test  = data2[["Age","Parch","Sex"]]

X_train.shape, Y_train.shape, X_test.shape
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Yy_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
Yy_pred
ids = data2['PassengerId']

output = pd.DataFrame({'PassengerId': ids, 'Survived': Yy_pred})
output.sample(5)
output.to_csv('ttggll.csv', index = False)
svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
Y_pred
ids = data2['PassengerId']

output = pd.DataFrame({'PassengerId': ids, 'Survived': Y_pred})
output.sample(5)
output.to_csv('tgl.csv', index = False)