# for understanding and analysis of data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# for making predictions and analysing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine_df = [train_df, test_df]

train_df.head()
train_df.describe()
train_df.info()
test_df.info()
sns.set_style('darkgrid')

fig, (ax1, ax2) = plt.subplots(1,2,sharey=True, figsize=(12,6))


# dataframe having true value for 'nan' values and vice-versa
null_bool_train = train_df.isnull() 
null_bool_test = test_df.isnull()

# heat map shows the label where there is no missing data
sns.heatmap(null_bool_train, cbar=False, yticklabels=False, ax=ax1)
sns.heatmap(null_bool_test, cbar=False, yticklabels=False, ax=ax2)

ax1.set_title('Training Data')
ax2.set_title('Testing Data')
for dataset in combine_df:
    dataset.drop('Cabin', inplace=True, axis=1)
train_df['Survived'].value_counts()
# 0 : not survived, 1 : Survived
ax = sns.countplot(x='Survived', data=train_df)
ax = sns.countplot(x = 'Survived', hue = 'Pclass', data = train_df) 
ax = sns.countplot(x = 'Survived', hue='Sex', data=train_df)
ax = sns.countplot(x = 'Survived', hue='Embarked', data=train_df)
plt.figure(figsize=(10,5))
ax = sns.FacetGrid(train_df, col='Survived')
ax.map(plt.hist, 'Age')
ax = sns.FacetGrid(train_df, row='Pclass', col='Sex')
ax.map(plt.hist, 'Age')
for df in combine_df:
    for pclass in [1,2,3]:
        for sex in ['male','female']:
            data_grp = df[(df['Sex'] == sex) & (df['Pclass'] == pclass)]
            average_age = data_grp[data_grp.isnull() == False]['Age'].mean()
            df.loc[(df.Age.isnull()) & (df.Sex == sex) & (df.Pclass == pclass), 'Age'] = average_age
            
    df['Age'] = df['Age'].astype(int)
ax = sns.distplot(train_df['Age'], bins=30, kde=False)
for df in combine_df:
    df.dropna(inplace=True)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))

sns.distplot(train_df['Fare'], bins=30, kde=False, ax=ax1)
ax1.set_title('Training Data')

sns.distplot(test_df['Fare'], bins=30, kde=False, ax=ax2)
ax2.set_title('Testing Data')

train_df[train_df['Fare'] > 400]
train_df.drop([258,679,737], axis=0, inplace=True)

fig,(ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

sns.distplot(train_df['SibSp'], kde=False, ax = ax1)
ax1.set_xlabel('Sibling or Spouse on board')

sns.distplot(train_df['Parch'], kde=False, ax = ax2)
ax2.set_xlabel('Parents or children on board')
for df in combine_df:
    df['Sex'] = df['Sex'].map({'female':1, 'male':0})# sex column
    df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})# embarked column
train_df.head()
test_df.head()
for df in combine_df:
    df.drop(['Name', 'Ticket'], axis=1, inplace=True)
XTrain = train_df.drop(['PassengerId','Survived'], axis=1)
yTrain = train_df['Survived']

columns = XTrain.columns
P_id = test_df['PassengerId']
scaler = StandardScaler()

XTrain = scaler.fit_transform(XTrain)

test_df.drop('PassengerId', axis=1, inplace=True)
test_data = scaler.fit_transform(test_df)
clf = LogisticRegression()

clf.fit(XTrain, yTrain)

clf.score(XTrain,yTrain)
X_train, X_test, y_train, y_test = train_test_split(XTrain, yTrain, test_size=0.33, random_state=42)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
# optional
X_train = sm.add_constant(X_train)
est = smf.Logit(y_train, X_train).fit()
coef_df = est.summary2().tables[1]
columns = columns.insert(0,'Intercept')
coef_df.index = columns
coef_df
# submission

y_pred = clf.predict(test_data)

submission = pd.DataFrame({'PassengerId':P_id, 'Survived':y_pred})

submission.to_csv('submission.csv',index=False)