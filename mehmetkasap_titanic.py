# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')
data.head()
data.info()
data.describe()
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True)
plt.show()
plt.figure(figsize=(15,7))
sns.countplot(data.Pclass, palette="icefire")
plt.show()
import warnings 
warnings.filterwarnings("ignore")

pclass1_survived = data[data.Pclass == 1][data.Survived == 1]
pclass1_dead = data[data.Pclass == 1][data.Survived == 0]

pclass2_survived = data[data.Pclass == 2][data.Survived == 1]
pclass2_dead = data[data.Pclass == 2][data.Survived == 0]

pclass3_survived = data[data.Pclass == 3][data.Survived == 1]
pclass3_dead = data[data.Pclass == 3][data.Survived == 0]
number_of_survived = np.array([pclass1_survived.PassengerId.count(), pclass2_survived.PassengerId.count(), pclass3_survived.PassengerId.count()])
number_of_dead = np.array([pclass1_dead.PassengerId.count(), pclass2_dead.PassengerId.count(), pclass3_dead.PassengerId.count()])
number_of_survived
number_of_dead
data_array = np.concatenate((number_of_survived, number_of_dead), axis=0)
#Creating pandas dataframe from numpy array
new_dataset = pd.DataFrame({'Dead': data_array[3:6],'Survived':data_array[0:3]})
new_dataset['Pclass'] = [1,2,3]
new_dataset['death_ratio'] = new_dataset.Dead / (new_dataset.Survived + new_dataset.Dead)
new_dataset
# OR USING GROUPBY 
pclass_group = data[['Survived', 'Pclass']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
pclass_group
data.head(1)
import warnings 
warnings.filterwarnings("ignore")

men_survived = data[data.Sex == 'male'][data.Survived == 1].PassengerId.count()
men_died = data[data.Sex == 'male'][data.Survived == 0].PassengerId.count()
female_survived = data[data.Sex == 'female'][data.Survived == 1].PassengerId.count()
female_died = data[data.Sex == 'female'][data.Survived == 0].PassengerId.count()
x_bar = [men_survived, men_died, female_survived, female_died]
x_bar
x_bar_df = pd.DataFrame({'men_survived': x_bar[0:1],'men_died':x_bar[1:2], 'female_survived':x_bar[2:3], 'female_died':x_bar[3:4]})
x_bar_df
x_bar_df['men_death_ratio'] = x_bar_df.men_died / (x_bar_df.men_died + x_bar_df.men_survived)
x_bar_df['female_death_ratio'] = x_bar_df.female_died / (x_bar_df.female_died + x_bar_df.female_survived)
x_bar_df
data = pd.read_csv('../input/train.csv')
data.Embarked.unique()
S_embarked = data[data.Embarked == 'S']
S_embarked_survival_rate = np.mean(S_embarked.Survived)*100
print('S_embarked_survival_rate: {} %'.format(S_embarked_survival_rate))
print('S_embarked_deat_rate: {} %'.format(100-S_embarked_survival_rate))
print('')

C_embarked = data[data.Embarked == 'C']
C_embarked_survival_rate = np.mean(C_embarked.Survived)*100
print('C_embarked_survival_rate: {} %'.format(C_embarked_survival_rate))
print('C_embarked_death_rate: {} %'.format(100-C_embarked_survival_rate))
print('')

Q_embarked = data[data.Embarked == 'Q']
Q_embarked_survival_rate = np.mean(Q_embarked.Survived)*100
print('Q_embarked_survival_rate: {} %'.format(Q_embarked_survival_rate))
print('Q_embarked_death_rate: {} %'.format(100-Q_embarked_survival_rate))
data = pd.read_csv('../input/train.csv')
data.head(1)
data.Parch.unique()
liste1 = []
for i in range(0,7):
    parch = data[data.Parch == i]
    parch_survival_percent = np.mean(parch.Survived)*100
    liste1.append(parch_survival_percent)
    
liste2 = []
for i in range(0,7):
    parch = data[data.Parch == i]
    count_i = parch.PassengerId.count()
    liste2.append(count_i)    
    
df2 = pd.DataFrame(liste2)
    
df1 = pd.DataFrame(liste1)
df1['Parch'] = np.arange(0,7)
df1['Survival_Percent'] = df1.iloc[:,0]
df1['Death Percent'] = 100-df1.Survival_Percent
df1 = df1.iloc[:,1:4]
df1['Count'] = df2.iloc[:,0]
df1
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(data, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(data, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(data, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand=False)

pd.crosstab(data['Title'], data['Sex'])
data = pd.read_csv('../input/train.csv')

# add title
data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand=False)
data.head(1)
data.Title.unique()
data['Title'] = [1 if i=='Mr' else 2 if i=='Mrs' else 3 if i=='Miss' else 4 if i=='Master'
                else 5 if i=='Don' else 6 if i=='Rev'else 7 if i=='Dr' else 8 if i=='Mme'
                else 9 if i=='Ms' else 10 if i=='Major' else 11 if i=='Lady' else 12 if i=='Sir'
                else 13 if i=='Mlle' else 14 if i=='Col' else 15 if i=='Capt' 
                else 16 if i=='Countess' else 17 for i in data.Title]
data_for_lg = data.drop(['PassengerId','Name', 'Cabin', 'Ticket'], axis=1)
data_for_lg['family_size'] = data.SibSp + data.Parch + 1

# drop the rows that has null values in order not to affect results
data_lg = data_for_lg.dropna()
data_lg.head()
x = data_lg.iloc[:, 1:]
x.head()
x['Title'] = x.Title / np.max(x.Title)

x['family_size'] = x.family_size / np.max(x.family_size)

x['Pclass'] = x.Pclass / np.max(x.Pclass)

x['Sex'] = [1 if i=='male' else 0 for i in x.Sex]

x['Age'] = (x.Age- np.min(x.Age))/(np.max(x.Age)- np.min(x.Age))

x['SibSp'] = x.SibSp / np.max(x.SibSp)

x['Parch'] = x.Parch / np.max(x.Parch)

x['Fare'] = (x.Fare- np.min(x.Fare))/(np.max(x.Fare)- np.min(x.Fare))

x['Embarked'] = [0.33 if i=='S' else 0.66 if i=='C' else 0.99 for i in x.Embarked]

y = data_lg.Survived

x.head()
x_copy = x
y_copy = y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.15, random_state=42)
x_train.head()
y_train.head()
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr.predict(x_test)
score_of_logistic_regression = lr.score(x_test,y_test)
print('score_of_logistic_regression: ', score_of_logistic_regression)
# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)
print('KNN score: ', knn.score(x_test, y_test))
neig = np.arange(1, 40)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))

# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
K = 1+test_accuracy.index(np.max(test_accuracy))
knn = KNeighborsClassifier(n_neighbors = K)
knn.fit(x_train,y_train)
print('KNN train score {} with K = {}'.format(knn.score(x_train, y_train), 1+test_accuracy.index(np.max(test_accuracy))))
print('KNN test score {} with K = {}'.format(knn.score(x_test, y_test), 1+test_accuracy.index(np.max(test_accuracy))))
# Confusion matrix with random forest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 9)
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('Confusion matrix: \n',cm)
print('Classification report: \n',classification_report(y_test,y_pred))
print('Random Forest train score: ', rf.score(x_train, y_train))
print('Random Forest test score: ', rf.score(x_test, y_test))
# LinearRegression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

# R^2 
print('Linear Regression train score: ',reg.score(x_train, y_train))
print('Linear Regression test score: ',reg.score(x_test, y_test))
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
Y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
acc_decision_tree_test = round(decision_tree.score(x_test, y_test) * 100, 2)

print('decision tree train score: ', acc_decision_tree)
print('decision tree test score: ', acc_decision_tree_test)
import warnings 
warnings.filterwarnings("ignore")
test = pd.read_csv('../input/test.csv')
test.head(2)
# add title
test['Title'] = test.Name.str.extract('([A-Za-z]+)\.', expand=False)
test.head(1)
test.Title.unique()
test['Title'] = [1 if i=='Mr' else 2 if i=='Mrs' else 3 if i=='Miss' else 4 if i=='Master'
                else 2 if i=='Ms' else 6 if i=='Col'else 7 if i=='Rev' else 8 if i=='Dr'
                else 9 for i in test.Title]
test.head(2)
data_for_lg = test.drop(['PassengerId','Name', 'Cabin', 'Ticket'], axis=1)
data_for_lg['family_size'] = test.SibSp + test.Parch + 1

# fill the rows that has null values in order not to affect results
data_lg = data_for_lg.fillna(1)
data_lg.head()
x = data_lg
x.head()
x['Title'] = x.Title / np.max(x.Title)
# x['Title'] = (x.Title- np.min(x.Title))/(np.max(x.Title)- np.min(x.Title))

x['family_size'] = x.family_size / np.max(x.family_size)
# x['family_size'] = (x.family_size- np.min(x.family_size))/(np.max(x.family_size)- np.min(x.family_size))

x['Pclass'] = x.Pclass / np.max(x.Pclass)
# x['Pclass'] = (x.Pclass- np.min(x.Pclass))/(np.max(x.Pclass)- np.min(x.Pclass))

x['Sex'] = [1 if i=='male' else 0 for i in x.Sex]

x['Age'] = (x.Age- np.min(x.Age))/(np.max(x.Age)- np.min(x.Age))

x['SibSp'] = x.SibSp / np.max(x.SibSp)
# x['SibSp'] = (x.SibSp- np.min(x.SibSp))/(np.max(x.SibSp)- np.min(x.SibSp))

x['Parch'] = x.Parch / np.max(x.Parch)
# x['Parch'] = (x.Parch- np.min(x.Parch))/(np.max(x.Parch)- np.min(x.Parch))

x['Fare'] = (x.Fare- np.min(x.Fare))/(np.max(x.Fare)- np.min(x.Fare))

x['Embarked'] = [0.33 if i=='S' else 0.66 if i=='C' else 0.99 for i in x.Embarked]

x.head()
y_test = knn.predict(x)
# y_test = rf.predict(x)
y_test.shape
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_test
    })
# submission.to_csv('../output/submission.csv', index=False)
submission.to_csv("Titanic_Mechmet_Kasap.csv",index=False)