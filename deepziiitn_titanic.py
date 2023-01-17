import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.preprocessing import Imputer

from mpl_toolkits.mplot3d import Axes3D 

plt.style.use('ggplot')

dataset = pd.read_csv('/kaggle/input/titanic-dataset-from-kaggle/train.csv') # load data from csv

test = pd.read_csv('/kaggle/input/titanic-dataset-from-kaggle/test.csv')

test.head()
# Plot histogram using seaborn

import seaborn as sns

plt.figure(figsize=(15,6))



sns.distplot(dataset.Age, bins =30)

X = dataset.iloc[:, :-1].values

total = dataset.isnull().sum().sort_values(ascending=False)

percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

f, ax = plt.subplots(figsize=(15, 6))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['Percent'])

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

missing_data.head()

total = dataset.isnull().sum().sort_values(ascending=False)

percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

f, ax = plt.subplots(figsize=(15, 6))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['Percent'])

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

missing_data.head()

dataset.Fare.fillna(dataset.Fare.mean(),inplace=True)

total = dataset.isnull().sum().sort_values(ascending=False)

percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

f, ax = plt.subplots(figsize=(15, 6))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['Percent'])

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

missing_data.head()

dummies = pd.get_dummies(dataset.Cabin)

merged=pd.concat([dataset,dummies],axis="columns")

merged
final=merged.drop(["Cabin","A10"],axis="columns")
datadict = pd.DataFrame(final.dtypes)

datadict
datadict['MissingVal'] = final.isnull().sum()

datadict

datadict['NUnique']=final.nunique()

datadict

datadict['Count']=final.count()



datadict = datadict.rename(columns={0:'DataType'})

datadict

final.describe(include=['object'])

final.describe(include=['number'])

final.Survived.value_counts(normalize=True)

fig, axes = plt.subplots(2, 4, figsize=(16, 10))

sns.countplot('Survived',data=final,ax=axes[0,0])

sns.countplot('Pclass',data=final,ax=axes[0,1])

sns.countplot('Sex',data=final,ax=axes[0,2])

sns.countplot('SibSp',data=final,ax=axes[0,3])

sns.countplot('Parch',data=final,ax=axes[1,0])

sns.countplot('Embarked',data=final,ax=axes[1,1])

sns.distplot(dataset['Fare'], kde=True,ax=axes[1,2])

sns.distplot(dataset['Age'].dropna(),kde=True,ax=axes[1,3])

figbi, axesbi = plt.subplots(2, 4, figsize=(16, 10))

final.groupby('Pclass')['Survived'].mean().plot(kind='barh',ax=axesbi[0,0],xlim=[0,1])

final.groupby('SibSp')['Survived'].mean().plot(kind='barh',ax=axesbi[0,1],xlim=[0,1])

final.groupby('Parch')['Survived'].mean().plot(kind='barh',ax=axesbi[0,2],xlim=[0,1])

final.groupby('Sex')['Survived'].mean().plot(kind='barh',ax=axesbi[0,3],xlim=[0,1])

final.groupby('Embarked')['Survived'].mean().plot(kind='barh',ax=axesbi[1,0],xlim=[0,1])

sns.boxplot(x="Survived", y="Age", data=final,ax=axesbi[1,1])

sns.boxplot(x="Survived", y="Fare", data=final,ax=axesbi[1,2])

sns.jointplot(x="Age", y="Fare", data=final);

import seaborn as sns



f, ax = plt.subplots(figsize=(10, 8))

corr = final.corr()

sns.heatmap(corr,

            mask=np.zeros_like(corr, dtype=np.bool), 

            cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)

dataset['Name_len']=dataset.Name.str.len()

dataset['Ticket_First']=dataset.Ticket.str[0]

dataset['FamilyCount']=dataset.SibSp+dataset.Parch

dataset['Cabin_First']=dataset.Cabin.str[0]

# Regular expression to get the title of the Name

dataset['title'] = dataset.Name.str.extract('\, ([A-Z][^ ]*\.)',expand=False)

dataset.title.value_counts().reset_index()

print((final.Fare == 0).sum())

final.Fare = final.Fare.replace(0, np.NaN)

# validate to see if there are no more zero values

print((final.Fare == 0).sum())



# keep the index

final[final.Fare.isnull()].index

final.Fare.median()

final.Fare.fillna(final.Fare.median(),inplace=True)

print((final.Age == 0).sum())

final.Age.fillna(final.Age.mean(),inplace=True)

final[final.Age.isnull()]

final.info()

final.columns

trainML = final.dropna()

trainML.isnull().sum()

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

X_Age = trainML[['Age']].values

y = trainML['Survived'].values

# Use the fit method to train

lr.fit(X_Age,y)

# Make a prediction

y_predict = lr.predict(X_Age)

y_predict[:10]

(y == y_predict).mean()

X_sex = pd.get_dummies(trainML['Sex']).values

y = trainML['Survived'].values

# Use the fit method to train

lr.fit(X_sex, y)

# Make a prediction

y_predict = lr.predict(X_sex)

y_predict[:10]

(y == y_predict).mean()

X_pclass = pd.get_dummies(trainML['Pclass']).values

y = trainML['Survived'].values

lr = LogisticRegression()

lr.fit(X_pclass, y)

# Make a prediction

y_predict = lr.predict(X_pclass)

y_predict[:10]

(y == y_predict).mean()

from sklearn.ensemble import RandomForestClassifier

X=trainML[['Age', 'SibSp', 'Parch','Fare']].values # Taking all the numerical values

y = trainML['Survived'].values

RF = RandomForestClassifier()

RF.fit(X, y)

# Make a prediction

y_predict = RF.predict(X)

y_predict[:10]

(y == y_predict).mean()
