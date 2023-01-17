import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



% matplotlib inline

% xmode plain



sns.set()

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/train.csv')

df_test =  pd.read_csv('../input/test.csv')
df.head()
df_test.shape
df.tail()
df_test.head()
df.describe()
df = df.drop(['PassengerId', 'Name'], axis = 1)

df_test = df_test.drop(['Name'], axis = 1)

df_test.shape
df.head()
sex = np.unique(df['Sex'], return_counts=True)

plt.bar(sex[0],sex[1],color=['blue','green'])

plt.ylabel('Number of individuals')

plt.xlabel('Sex')
ages = np.unique(df[np.isfinite(df['Age'])]['Age'], return_counts=True)

plt.bar(ages[0],ages[1])

plt.ylabel('Number of people')

plt.xlabel('Ages')
females = df[df['Sex'] == 'female']

males = df[df['Sex'] == 'male']

# print(females)



female_ages = np.unique(females[np.isfinite(df['Age'])]['Age'], return_counts=True)

male_ages = np.unique(males[np.isfinite(df['Age'])]['Age'], return_counts=True)



fig = plt.figure(figsize=(16, 9))

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)





ax1.bar(female_ages[0],female_ages[1], color='blue')

ax1.set_ylabel('Number of people')

ax1.set_xlabel('Age')



ax2.bar(male_ages[0], male_ages[1], color='green')

ax2.set_ylabel('Number of people')

ax2.set_xlabel('Age')
g = sns.FacetGrid(df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
df.head()
df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)

df_test['Sex'] = df_test['Sex'].apply(lambda x: 1 if x == 'male' else 0)
fig = plt.figure(figsize=(16, 9))

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)



females = df[df['Sex'] == 0]

setFemales = np.unique(females['Survived'].values, return_counts=True) 

ax1 = sns.barplot(setFemales[0], setFemales[1],ax = ax1,)

ax1.set(xlabel='Survived', ylabel='number')

ax1.set_title('Females')



females = df[df['Sex'] == 1]

setFemales = np.unique(females['Survived'].values, return_counts=True) 

ax2 = sns.barplot(setFemales[0], setFemales[1],ax = ax2)

ax2.set(xlabel='Survived', ylabel='number')

ax2.set_title('Males')
def generateCompData(feature):



    groupedByClass = df[['Survived', feature]].groupby([feature])

    survivedPclass = []

    diedPclass = []



    for grpname in sorted(df[feature].unique()):    

        grp = groupedByClass.get_group(grpname)

        

        survivedPclass.append(np.sum(grp.Survived == 1))

        diedPclass.append(np.sum(grp.Survived == 0))

        

    index = np.arange(1,len(diedPclass) + 1)

    

    return survivedPclass, diedPclass, index
def generateCompPlot(feature, survived, died, index, width = 0.3):

    

    plt.bar(index, diedPclass, width=0.3)

    plt.bar(index + 0.3, survivedPclass, width=0.3)

    plt.xticks(index)

    plt.xlabel(feature)

    plt.ylabel('Number of People')

    plt.legend(['Died','Survided'], ncol=2, loc='upper left');

    
survivedPclass, diedPclass, index = generateCompData("Pclass")

generateCompPlot("Pclass", survivedPclass, diedPclass, index)

# plt.bar(index, diedPclass)
g = sns.FacetGrid(df, col='Survived', row='Pclass')

g.map(plt.hist, 'Age', bins=20)
plt.plot(df.index.values, df.Fare.values)
df = df[df['Fare'] < 300]

df['Fare'].max()

np.linspace(0, df['Fare'].max(), 4, endpoint=True)
def class_fare(x):

    

    linear_space = np.linspace(0, 263, 4, endpoint=True)

    

    if x  < linear_space[1]:

        return 0

    elif x < linear_space[2]:

        return 1

    elif x < linear_space[3]:

        return 2

df['Fare'] = df['Fare'].apply(class_fare)

df_test['Fare'] = df_test['Fare'].apply(class_fare)

df_test.shape
g = sns.FacetGrid(df, col='Survived', row='Fare')

g.map(plt.hist, 'Age', bins=20)
df['Embarked'] = df['Embarked'].map({'C':1, 'Q':2, 'S':3})

df_test['Embarked'] = df_test['Embarked'].map({'C':1, 'Q':2, 'S':3})
g = sns.FacetGrid(df, col='Survived', row='Embarked')

g.map(plt.hist, 'Age', bins=20)
sns.heatmap( df.corr() , center = 0 )
df.head()
df['Age'] = df['Age'].fillna(df['Age'].mean())

df['Fare'] =  df['Fare'].fillna(df['Fare'].mean())

df['Embarked'] = df['Embarked'].fillna(method='ffill')



df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())

df_test['Fare'] =  df_test['Fare'].fillna(df_test['Fare'].mean())

df_test['Embarked'] = df_test['Embarked'].fillna(method='ffill')
df.isnull().any()
df = df.drop(columns=['Ticket', 'Cabin'])

df_test = df_test.drop(columns=['Ticket', 'Cabin'])
print(df.isnull().any())

print(df_test.isnull().any())
x_train = df.values[:, 1:]

y_train = df.values[:, 0]

x_test = df_test.values[:, 1:]
from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
randForest = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=22)

randForest.fit(x_train, y_train)
y_predict = randForest.predict(x_train)

confMatrix = confusion_matrix(y_predict, y_train)
acc_rf = randForest.score(x_train, y_train)

acc_rf
sns.set(font_scale=0.9)

sns.heatmap(confMatrix, annot=True,annot_kws={"size": 16}, center=5)
NN = MLPClassifier(activation='relu', solver='adam', random_state=22)

NN.fit(x_train, y_train)
acc_nn = NN.score(x_train, y_train)

df_test.head()
y_predict = NN.predict(x_train)

confMatrix = confusion_matrix(y_predict, y_train)

sns.set(font_scale=0.9)

sns.heatmap(confMatrix, annot=True,annot_kws={"size": 16}, center=-1)
svc = SVC(gamma='auto')

svc.fit(x_train, y_train)
acc_svc = svc.score(x_train, y_train)
y_predict = svc.predict(x_train)

confMatrix = confusion_matrix(y_predict, y_train)

sns.set(font_scale=0.9)

sns.heatmap(confMatrix, annot=True,annot_kws={"size": 16}, center=-1)
pd.DataFrame([[acc_rf], [acc_nn], [acc_svc]], columns=[ 'Accuracy'],  index=['Random Forest', 'Multilayer Percentron', 'Support Vector Machine'])
y_predict = NN.predict(x_test)
submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": y_predict })

submission['Survived'] = submission['Survived'].apply(lambda x: int(x))

submission.head()

print(submission.head())

submission.to_csv('./submission.csv', index=False)