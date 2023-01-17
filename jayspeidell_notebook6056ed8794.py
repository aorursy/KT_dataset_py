import pandas as pd

import numpy as np

import os

import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt
os.chdir('../input')

df_train = pd.read_csv('train.csv', header=0, index_col=0, sep=',')



# create copy just for EDA

df_eda = df_train.copy()

df_test = pd.read_csv('test.csv', header=0, index_col=0, sep=',')



columns = df_eda.columns
def null_percentage(column):

    df_name = column.name

    nans = np.count_nonzero(column.isnull().values)

    total = column.size

    frac = nans / total

    perc = int(frac * 100)

    print('%d%% of values or %d missing from %s column.' % (perc, nans, df_name))



for col in columns:

    null_percentage(df_eda[col])
df_eda.head(10)
map_sex = {'male': 1, 'female': 0}

df_eda.Sex = df_eda.Sex.replace(map_sex)
df_eda['age_known'] = df_eda.Age.replace(np.nan, 0).astype(int)

df_eda['age_known'][df_eda['age_known'] != 0] = 1

df_eda.age_known.value_counts()
df_eda['cabin_known'] = df_eda.Cabin.replace(np.nan, 0)

df_eda['cabin_known'][df_eda['cabin_known'] != 0] = 1

df_eda.cabin_known.value_counts()
# generate a correlation matrix and build a heatmap

plt.figure('heatmap')

_ = sns.heatmap(df_eda.corr(), vmax=0.6, square=True, annot=True)

plt.show()
tab = pd.crosstab(df_train['Sex'], df_train['Survived'])

tab.plot(kind='bar', stacked='true', color=['red','blue'], grid=False)

print(tab)

plt.show()
plt.figure('age distribution', figsize=(18,12))

plt.subplot(411)

ax = sns.distplot(df_eda.Age.dropna().values, bins=range(0,81,1), kde=False, axlabel='Age')

plt.subplot(412)

sns.distplot(df_eda[df_eda['Survived'] == 1].Age.dropna().values, bins = range(0, 81, 1), 

             color='blue', kde=False)

sns.distplot(df_eda[df_eda['Survived'] == 0].Age.dropna().values, bins = range(0, 81, 1), 

             color='red', kde=False, axlabel='All survivors by age')

plt.subplot(413)

sns.distplot(df_eda[(df_eda['Sex']==0) & (df_eda['Survived'] == 1)].Age.dropna().values, bins = range(0, 81, 1), 

             color='blue', kde=False)

sns.distplot(df_eda[(df_eda['Sex']==0) & (df_eda['Survived'] == 0)].Age.dropna().values, bins = range(0, 81, 1), 

             color='red', kde=False, axlabel='Female survivors by age')

plt.subplot(414)

sns.distplot(df_eda[(df_eda['Sex']==1) & (df_eda['Survived'] == 1)].Age.dropna().values, bins = range(0, 81, 1), 

             color='blue', kde=False)

sns.distplot(df_eda[(df_eda['Sex']==1) & (df_eda['Survived'] == 0)].Age.dropna().values, bins = range(0, 81, 1), 

             color='red', kde=False, axlabel='Male survivors by age')

plt.show()



tab = pd.crosstab(df_eda['age_known'], df_eda['Survived'])

print(tab)

tab.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

plt.show()
tab = pd.crosstab(df_eda['Survived'], df_eda['Pclass'])

print(tab)

tab.plot(kind='bar', stacked=True, color=['darkgreen', 'green', 'lightgreen'], grid=False)

plt.show()

#tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", color=['darkgreen', 'green', 'lightgreen'], stacked=True)

#plt.show()

tab = pd.crosstab(df_eda['cabin_known'], df_eda['Survived'])

print(tab)

tab.plot(kind='bar', stacked=True, color=['red', 'blue'], grid=False)

plt.show()
sns.violinplot(x='Pclass', y='Fare', hue='Survived', data=df_eda, split=True)

plt.hlines([0,512], xmin=-1, xmax=3)

plt.show()
tab = pd.crosstab(df_eda['Embarked'], df_eda['Survived'])

print(tab)

tab.plot(kind='bar', stacked=True, color=['red', 'blue'], grid=False)

plt.show()
tab = pd.crosstab(df_eda['Embarked'], df_eda['Pclass'])

print(tab)
for col in columns[1:]:

    null_percentage(df_test[col])
print(df_test[df_test['Fare'].isnull()])
df_train['Embarked'][df_train['Embarked'] == np.nan] = 'S'

df_test['Fare'][df_test['Name'] == 'Storey, Mr. Thomas'] = df_test['Fare'][df_test['Pclass'] == 3].median()



# select target values

y_targets = df_train.iloc[:,0].values



# combine for transformation

df_train['train'] = 1

df_test['train'] = 0

df = pd.concat([df_train, df_test], ignore_index=False, axis=0)



# select the columns to persist after transforming

train_cols = ['Pclass', 'Sex', 'age_known', 'cabin_known', 'Young', 'Fare', 

             'Embarked_Q', 'Embarked_S', 'train']



map_sex = {'male': 1, 'female': 0}

df.Sex = df.Sex.replace(map_sex)

df['age_known'] = df.Age.replace(np.nan, 0).astype(int)

df['age_known'][df['age_known'] != 0] = 1

    

df['cabin_known'] = df.Cabin.replace(np.nan, 0)

df['cabin_known'][df['cabin_known'] != 0] = 1



young_bool = (df.age_known == 1) & (df.Age < 9)

df['Young'] = young_bool.astype(int)

    

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)



df = df[train_cols]



# split back into training and test set after transforming

df_train = df[df['train'] == 1].drop(['train'], axis = 1)

df_test = df[df['train'] == 0].drop(['train'], axis = 1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_train, y_targets, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))



from sklearn.model_selection import cross_val_score

cvs = cross_val_score(classifier, X_test, y_test, cv=10)

print(cvs)
y_pred_test = classifier.predict(df_test)

df_test['Survived'] = y_pred_test