import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plot data

import sklearn #scikit-learn library, where the magic happens!

import seaborn as sns # beautiful graphs

import re
df_titanic = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')



# Lowercase content

# df_titanic = df_titanic.applymap(lambda s:s.lower() if isinstance(s, str) else s)



df_titanic = df_titanic.drop(columns=['PassengerId'])
df_titanic
df_titanic['Female'] = (df_titanic['Sex']=='female')

df_titanic['Male'] = (df_titanic['Sex']=='male')



df_test['Female'] = (df_test['Sex']=='female')

df_test['Male'] = (df_test['Sex']=='male')
# Compute the correlation matrix

corr = df_titanic.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(230, 20, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, mask=mask, vmax=.3, center=0, annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={"size": 10})
sns.set(font_scale=2)

plt.figure(figsize=(10, 5))

ax = sns.boxplot(x='Age', orient='h',data=df_titanic,fliersize=8)
sns.set(font_scale=2)

plt.figure(figsize=(10, 5))

ax = sns.boxplot(x='Age',y='Survived', orient='h',data=df_titanic,fliersize=8)
sns.set(font_scale=2)

plt.figure(figsize=(10, 5))

ax = sns.boxplot(x='Fare', orient='h',data=df_titanic,fliersize=8)
sns.set(font_scale=2)

plt.figure(figsize=(10, 5))

ax = sns.boxplot(x='Fare',y='Survived', orient='h',data=df_titanic,fliersize=8)
sns.set(font_scale=2)

plt.figure(figsize=(10, 5))

ax = sns.boxplot(x='Survived', y='Age',hue='Pclass',orient='v',data=df_titanic,fliersize=8)
ax = sns.barplot(x="Sex", y="Survived", data=df_titanic)
ax = sns.barplot(x="Pclass", y="Survived", hue="Sex", data=df_titanic)
sns.set(font_scale=2)

plt.figure(figsize=(12, 8))

sns.distplot(df_titanic[df_titanic['Survived']==0]['Age'], color='red', label='0')

sns.distplot(df_titanic[df_titanic['Survived']==1]['Age'], color='skyblue', label='1')

plt.legend()
ax = sns.barplot(x="Embarked", y="Survived", data=df_titanic)
df_titanic.loc[df_titanic['Age'] <= 10, 'Age Group'] = 1

df_titanic.loc[(df_titanic['Age'] > 10) & (df_titanic['Age'] <= 22) , 'Age Group'] = 2

df_titanic.loc[(df_titanic['Age'] > 22) & (df_titanic['Age'] <= 40), 'Age Group'] = 3

df_titanic.loc[df_titanic['Age'] > 40, 'Age Group'] = 4



df_test.loc[df_test['Age'] <= 10, 'Age Group'] = 1

df_test.loc[(df_test['Age'] > 10) & (df_test['Age'] <= 22) , 'Age Group'] = 2

df_test.loc[(df_test['Age'] > 22) & (df_test['Age'] <= 40), 'Age Group'] = 3

df_test.loc[df_test['Age'] > 40, 'Age Group'] = 4
df_titanic['Group'] = df_titanic['SibSp']+df_titanic['Parch']+1

df_test['Group'] = df_test['SibSp']+df_test['Parch']+1
df_titanic['Fare Group'] = df_titanic['Fare'] / df_titanic['Group']

df_test['Fare Group'] = df_test['Fare'] / df_test['Group']
df_titanic['Name Lenght'] = df_titanic['Name'].str.len()

df_test['Name Lenght'] = df_test['Name'].str.len()
df_titanic['Deck'] = df_titanic['Cabin'].str.extract(r'([A-Za-z])?')

df_titanic['Room'] = df_titanic['Cabin'].str.extract(r'([0-9]+)')



df_test['Deck'] = df_test['Cabin'].str.extract(r'([A-Za-z])?')

df_test['Room'] = df_test['Cabin'].str.extract(r'([0-9]+)')



regex = re.compile('\s*(\w+)\s*')



df_titanic['Cabin'] = df_titanic['Cabin'].fillna('0')

df_titanic['CabinNum'] = df_titanic['Cabin']

df_titanic['CabinNum'] = df_titanic['CabinNum'].apply(lambda x : len(regex.findall(x)))



df_test['Cabin'] = df_test['Cabin'].fillna('0')

df_test['CabinNum'] = df_test['Cabin']

df_test['CabinNum'] = df_test['CabinNum'].apply(lambda x : len(regex.findall(x)))
from sklearn import preprocessing



df_titanic['Deck'].fillna('Z', inplace=True)

df_test['Deck'].fillna('Z', inplace=True)



df_titanic['Room'].fillna(0, inplace=True)

df_test['Room'].fillna(0, inplace=True)



df_titanic['Embarked'].fillna('S', inplace=True)

df_test['Embarked'].fillna('S', inplace=True)



label_port = preprocessing.LabelEncoder()

label_port.fit(['C', 'Q', 'S'])

df_titanic['Embarked'] = label_port.transform(df_titanic['Embarked'])

df_test['Embarked'] = label_port.transform(df_test['Embarked'])



label_deck = preprocessing.LabelEncoder()

label_deck.fit(['A', 'B', 'C','D','E','F','G','T','Z'])

df_titanic['Deck'] = label_deck.transform(df_titanic['Deck'])

df_test['Deck'] = label_deck.transform(df_test['Deck'])



cabin_port = preprocessing.LabelEncoder()

cabin_port.fit(pd.concat([df_titanic['Cabin'], df_test['Cabin']],join='inner',sort=False))

df_titanic['Cabin'] = cabin_port.transform(df_titanic['Cabin'])

df_test['Cabin'] = cabin_port.transform(df_test['Cabin'])
# Compute the correlation matrix

corr = df_titanic.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(230, 20, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, mask=mask, vmax=.3, center=0, annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={"size": 10})
print('Train:')

print(df_titanic.isnull().sum())

print('Test:')

print(df_test.isnull().sum())
df_test['Fare'].fillna(0, inplace=True)

print(df_test[df_test['Fare']==0])

print('\n')



df_test.at[152,'Fare'] = df_titanic[df_titanic['Pclass']==3]['Fare'].mean()

df_test.at[266,'Fare'] = df_titanic[df_titanic['Pclass']==1]['Fare'].mean()

df_test.at[372,'Fare'] = df_titanic[df_titanic['Pclass']==1]['Fare'].mean()



df_test['Fare Group'] = df_test['Fare'] / df_test['Group']



df_titanic.loc[(df_titanic['Fare'] < 8),'Fare Cat'] = 0

df_titanic.loc[(df_titanic['Fare'] >= 8) & (df_titanic['Fare'] < 16),'Fare Cat'] = 1

df_titanic.loc[(df_titanic['Fare'] >= 16) & (df_titanic['Fare'] < 30),'Fare Cat'] = 2

df_titanic.loc[(df_titanic['Fare'] >= 30) & (df_titanic['Fare'] < 45),'Fare Cat'] = 3

df_titanic.loc[(df_titanic['Fare'] >= 45) & (df_titanic['Fare'] < 80),'Fare Cat'] = 4

df_titanic.loc[(df_titanic['Fare'] >= 80) & (df_titanic['Fare'] < 160),'Fare Cat'] = 5

df_titanic.loc[(df_titanic['Fare'] >= 160) & (df_titanic['Fare'] < 270),'Fare Cat'] = 6

df_titanic.loc[(df_titanic['Fare'] >= 270),'Fare Cat'] = 7



df_test.loc[(df_test['Fare'] < 8),'Fare Cat'] = 0

df_test.loc[(df_test['Fare'] >= 8) & (df_test['Fare'] < 16),'Fare Cat'] = 1

df_test.loc[(df_test['Fare'] >= 16) & (df_test['Fare'] < 30),'Fare Cat'] = 2

df_test.loc[(df_test['Fare'] >= 30) & (df_test['Fare'] < 45),'Fare Cat'] = 3

df_test.loc[(df_test['Fare'] >= 45) & (df_test['Fare'] < 80),'Fare Cat'] = 4

df_test.loc[(df_test['Fare'] >= 80) & (df_test['Fare'] < 160),'Fare Cat'] = 5

df_test.loc[(df_test['Fare'] >= 160) & (df_test['Fare'] < 270),'Fare Cat'] = 6

df_test.loc[(df_test['Fare'] >= 270),'Fare Cat'] = 7
from sklearn.neighbors import KNeighborsClassifier



df_titanic_drop = df_titanic.drop(columns=['Cabin']).dropna()

X_train_age = df_titanic_drop.drop(columns=['Survived','Name','Sex','Age','Age Group','Ticket'])

y_train_age = df_titanic_drop['Age Group'].dropna()



age_classifier = KNeighborsClassifier(n_neighbors=3)

age_classifier.fit(X_train_age, y_train_age)



X_test_age = df_test[df_test['Age Group'].isnull()].drop(columns=['PassengerId','Name','Sex','Age','Age Group','Ticket','Cabin'])

y_test_age = age_classifier.predict(X_test_age)

y_test_age



X_train_age = df_titanic[df_titanic['Age Group'].isnull()].drop(columns=['Survived','Name','Sex','Age','Age Group','Ticket','Cabin'])

y_train_age = age_classifier.predict(X_train_age)

y_train_age
df_test.loc[(df_test['Age Group'].isnull()),'Age Group'] = y_test_age

df_titanic.loc[(df_titanic['Age Group'].isnull()),'Age Group'] = y_train_age
mean_age_1 = df_titanic[df_titanic['Age Group']==1]['Age'].mean()

mean_age_2 = df_titanic[df_titanic['Age Group']==2]['Age'].mean()

mean_age_3 = df_titanic[df_titanic['Age Group']==3]['Age'].mean()

mean_age_4 = df_titanic[df_titanic['Age Group']==4]['Age'].mean()



df_titanic.loc[(df_titanic['Age'].isnull()) & (df_titanic['Age Group']==1),'Age'] = mean_age_1

df_titanic.loc[(df_titanic['Age'].isnull()) & (df_titanic['Age Group']==2),'Age'] = mean_age_2

df_titanic.loc[(df_titanic['Age'].isnull()) & (df_titanic['Age Group']==3),'Age'] = mean_age_3

df_titanic.loc[(df_titanic['Age'].isnull()) & (df_titanic['Age Group']==4),'Age'] = mean_age_4



df_test.loc[(df_test['Age'].isnull()) & (df_test['Age Group']==1),'Age'] = mean_age_1

df_test.loc[(df_test['Age'].isnull()) & (df_test['Age Group']==2),'Age'] = mean_age_2

df_test.loc[(df_test['Age'].isnull()) & (df_test['Age Group']==3),'Age'] = mean_age_3

df_test.loc[(df_test['Age'].isnull()) & (df_test['Age Group']==4),'Age'] = mean_age_4
X_train = df_titanic.drop(columns=['Survived','Name','Sex','Age','Ticket','Fare','Cabin','Fare Group'])

y_train = df_titanic['Survived']



X_test = df_test.drop(columns=['PassengerId','Name','Sex','Age','Ticket','Fare','Cabin','Fare Group'])
print('X_train:')

print(X_test.isnull().sum())
from sklearn.ensemble import RandomForestClassifier



tree = RandomForestClassifier(max_depth = 10, min_samples_split = 4, n_estimators = 500, random_state=0)

tree.fit(X_train, y_train)



y_test = pd.Series(tree.predict(X_test))



df_final = pd.concat([df_test['PassengerId'], y_test], axis=1, sort=False)

df_final = df_final.rename(columns={0:"Survived"})

df_final.to_csv(r'titanic_random_forrest.csv', index = False)