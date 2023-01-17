# Loading some of the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib notebook
import os
for dirname, _, filenames in os.walk('kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Loading the datasets used here
train_data = pd.read_csv('/kaggle/input/train-testtt/train_titanic.txt')
test_data = pd.read_csv('/kaggle/input/train-testtt/test_titanic.txt')
y = train_data['Survived']
# Reading the above data
print(y, train_data, test_data)
# Size of the datasets


train_data.shape, test_data.shape
# Data description
train_data.describe(), test_data.describe()
# Percentage number of those who survived by their sex
sexes = train_data.groupby('Sex').mean()
sexes['Survived']*100
# Percentage number of those who survived by their class
sexes = train_data.groupby('Pclass').mean()
sexes['Survived']*100
# Total number of people who survived and those who didn't make it labelled by 1 and 0 respectively 
train_data.Survived.value_counts()
# Percentage number of those who survived by SibSp and Unique SibSp_counts
KK = train_data.groupby('SibSp').mean()
KKK = KK['Survived']*100
SibSp_counts = train_data['SibSp'].value_counts()
print(KKK, SibSp_counts)
# Percentage number of those who survived by Parch and Unique Parch_counts
PP = train_data.groupby('Parch').mean()
PPP = PP['Survived']*100
Parch_counts = train_data['Parch'].value_counts()
print(PPP, Parch_counts)

plt.figure(figsize=(5,4))
plt.style.use('seaborn-notebook')
sns.countplot(x='Survived', hue='Sex', data=train_data)
plt.title('Countplot showing Survivors by their sex')
plt.show()
plt.figure()
plt.style.use('seaborn-dark')
sns.distplot(train_data['Age'], bins=50, kde=True, color='b')
plt.title('Age distribution amongst the train_dataset')
plt.show()

plt.style.use('seaborn-deep')
facet = sns.FacetGrid(data=train_data, col='Survived', height=3, aspect=4)
facet.map(sns.distplot, 'Age', kde=True, bins=50)
plt.title('Plots displaying age distribution among Survivors')
plt.show()
Relationship = train_data.corr()
plt.figure()
sns.heatmap(data=Relationship, annot=True, cmap='Blues')
plt.title('Relationship between different varaiables in our train_data')
plt.show()
def family(data):
    data['family_tog'] = data['Parch'] + data['SibSp']
    data = data.drop(['SibSp', 'Parch'], axis=1)
    return data
train_data = family(train_data)
test_data = family(test_data)
means = train_data.groupby('family_tog')
means.Survived.mean()*100
train_data['family_tog'] = train_data['family_tog'].apply(lambda x: 'Travelled alone' if x==0 else x)
train_data['family_tog'] = train_data['family_tog'].apply(lambda x: 'Travelled with small family' if (x==1 or x==2 or x==3) else x)
train_data['family_tog'] = train_data['family_tog'].apply(lambda x: 'Travelled with slightly big family' if (x==4 or x==5 or x==6) else x)
train_data['family_tog'] = train_data['family_tog'].apply(lambda x: 'Travelled with big family' if (x==7 or x==10) else x)

test_data['family_tog'] = test_data['family_tog'].apply(lambda x: 'Travelled alone' if x==0 else x)
test_data['family_tog'] = test_data['family_tog'].apply(lambda x: 'Travelled with small family' if (x==1 or x==2 or x==3) else x)
test_data['family_tog'] = test_data['family_tog'].apply(lambda x: 'Travelled with slightly big family' if (x==4 or x==5 or x==6) else x)
test_data['family_tog'] = test_data['family_tog'].apply(lambda x: 'Travelled with big family' if (x==7 or x==10) else x)

print(train_data['family_tog'].value_counts(), test_data['family_tog'].value_counts())
# Pie_Chart 
# More visualizations on the data
plt.figure()
plt.style.use('seaborn-muted')
plt.pie(train_data.Sex.value_counts(), explode=[0.02, 0.02], labels= ['Male', 'Female'], autopct='%1.1f%%', colors=['b','g'])
plt.title('Pie Chart showing distribution of men and women in the train_data')
plt.show()

plt.figure()
plt.style.use('seaborn-paper')
sexes = train_data.groupby('Sex').mean()
plt.pie(sexes.Survived, labels= ['Male', 'Female'], autopct='%1.1f%%', colors=['red','purple'])
plt.title('Pie Chart showing distribution of men and women who survived')
plt.show()

plt.figure()
plt.style.use('Solarize_Light2')
sexes = train_data.groupby('Embarked').mean()
plt.pie(sexes.Survived, labels= ['C', 'Q', 'S'], autopct='%1.1f%%', colors=['orange','blue', 'magenta'])
plt.title('Pie Chart showing passenger distribution at different Ports of embarkation')
plt.show()

plt.figure()
plt.style.use('seaborn-paper')
plt.pie(train_data['family_tog'].value_counts(), labels= ['Travelled alone', 'Travelled with small family', 'Travelled with slightly big family', 'Travelled with big family'], autopct='%1.1f%%', colors=['b','g', 'brown', 'grey'])
plt.title('Pie Chart showing distribution of men and women in the train_data')
plt.show()
train_data['Fare'].nunique()
train_data['Fare'].value_counts()
# Price variations at the different ports of embarkation
plt.style.use('seaborn-paper')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,8))
ax1.hist(train_data.Fare[train_data.Embarked == 'C'], bins=50, alpha=0.5)
ax1.set_yscale('log')
ax1.set_title('Price variations for those who embarked at Cherbourg')
plt.show()
plt.xlabel('FARE', color='black')
plt.tight_layout(pad=1.50)

plt.style.use('seaborn-paper')
ax2.hist(train_data.Fare[train_data.Embarked == 'Q'], bins=50, alpha=0.5, color='g')
ax2.set_yscale('log')
ax2.set_title('Price variations for those who embarked at Queensland')
plt.show()
plt.tight_layout(pad=1.50)

plt.style.use('seaborn-paper')
ax3.hist(train_data.Fare[train_data.Embarked == 'S'], bins=50, alpha=0.5, color='r')
ax3.set_yscale('log')
ax3.set_title('Price variations for those who embarked at Southampton')
plt.show()
train_data['Name'] = train_data['Name'].apply(lambda x: x.split('.')[0].split(',')[1])
test_data['Name'] = test_data['Name'].apply(lambda x: x.split('.')[0].split(',')[1])
train_data['Name'].value_counts(), test_data['Name'].value_counts()
plt.figure()
plt.style.use('seaborn-dark')
sns.countplot(x='Name', hue='Survived', data=train_data, palette='colorblind')
plt.xticks(rotation=60)
plt.title('A countplot showing name titles and their survival counts', color='g')
plt.show()
plt.legend(loc='upper right')
plt.tight_layout(pad=1.50)
title = pd.DataFrame(train_data.groupby('Name')['Survived'].agg('count').sort_values(ascending=False))
plt.figure()
plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize'] = (8,6)
sns.barplot(x=title.Survived, y=title.index, data=title, palette='colorblind')
plt.gca().set_title('Bar plot showing Name title counts', fontsize=(10))
plt.show()
train_data['Cabin'].value_counts()
train_data['Ticket'].value_counts()
train_data['Cabin'].isnull().sum(), test_data['Cabin'].isnull().sum()
train_data['Cabin'].fillna('Reserved', inplace=True), test_data['Cabin'].fillna('Reserved', inplace=True)
train_data
train_data['Cabin'].value_counts()
train_data['deck'] = train_data['Cabin'].str.replace('([0-9\s])+', '')
test_data['deck'] = test_data['Cabin'].str.replace('([0-9\s])+', '')
test_data['deck'].value_counts(), train_data['deck'].value_counts()
def number_of_cabins(row):
    if len(row.deck) > 1:
       row['Cabin'] = len(row.deck)
    elif row.deck == 'Reserved':
        row['Cabin'] = 0
    else:
        row['Cabin'] = 1
    return row
train_data = train_data.apply(number_of_cabins, axis=1)
test_data = test_data.apply(number_of_cabins, axis=1)
train_data['Cabin'].value_counts(), test_data['Cabin'].value_counts()
train_data['deck'] = train_data['deck'].apply(lambda x: x[0] if x != 'Reserved' else x)
test_data['deck'] = test_data['deck'].apply(lambda x: x[0] if x != 'Reserved' else x)
train_data['deck'].value_counts(), test_data['deck'].value_counts()
for title in train_data[train_data.Age.isna()].Name.value_counts().index:
    mean_age = train_data.groupby('Name').mean().T[title].Age
    mean_age_list = train_data[train_data.Name == title].Age.fillna(mean_age)
    train_data.update(mean_age_list)
train_data.isnull().sum()
for title in test_data[test_data.Age.isnull()].Name.value_counts().index:
    mean_age = test_data.groupby('Name').mean().T[title].Age
    mean_age_list = test_data[test_data.Name == title].Age.fillna(mean_age)
    test_data.update(mean_age_list)
test_data.Age.isna().sum()
test_data['Age'] = test_data.Age.fillna(test_data.Age.mean())
test_data.Age.isnull().sum()
most_paid_fare = test_data.Fare.value_counts()
test_data.Fare = test_data.Fare.fillna(most_paid_fare.index[0])
most_frequent_port_of_embarkation = train_data.Embarked.value_counts()
port = most_frequent_port_of_embarkation
port
train_data.Embarked = train_data.Embarked.fillna(port.index[0])
train_data.drop(['PassengerId','Survived', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(['PassengerId','Ticket', 'Cabin'], axis=1, inplace=True)
train_data
# Using OneHotEncoder to deal with categorical features to create dummies
from sklearn.preprocessing import OneHotEncoder
Enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
features = ['Pclass', 'Name', 'Sex', 'Age', 'Fare', 'Embarked', 'family_tog', 'deck']
Enc_train_cols = pd.DataFrame(Enc.fit_transform(train_data[features]))
Enc_test_cols = pd.DataFrame(Enc.transform(test_data[features]))

Enc_train_cols.index = train_data.index
Enc_test_cols.index = test_data.index

Trn = train_data.drop(features, axis=1)
Ten = test_data.drop(features, axis=1)

train_data = pd.concat([Trn, Enc_train_cols], axis=1)
test_data = pd.concat([Ten, Enc_test_cols], axis=1)
train_data.shape, test_data.shape
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
steps = [('scaler', StandardScaler())]
pipe = Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier())])
param_grid={'classifier': [RandomForestClassifier()], 'classifier__n_estimators':[500], 'classifier__max_features':[0.25], 'classifier__max_depth':[8], 'classifier__criterion':['entropy']}
clf = GridSearchCV(pipe, param_grid=param_grid, cv=10, verbose=True, n_jobs=-1)
best_clf = clf.fit(train_data, y)
score = clf.best_score_
params = clf.best_params_
print('Best score : {}'.format(score))
print('Best parameters : {}'.format(params))
final_results = clf.predict(test_data)
final_results
test_data = pd.read_csv('/kaggle/input/train-testtt/test_titanic.txt')
Final_solution = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': final_results})
Final_solution
Final_solution.to_csv('Submission.csv', index=False)


