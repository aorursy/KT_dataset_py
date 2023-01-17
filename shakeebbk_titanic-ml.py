import pandas as pd

import numpy as np
train_df = pd.read_csv('../input/titanic/train.csv')

validation_df = pd.read_csv('../input/titanic/test.csv')

train_df.head()
bins = pd.IntervalIndex.from_tuples([(0, 16), (17, 32), (33, 48), (49, 150)])

exploration_df = train_df

exploration_df['Age_bin'] = pd.cut(train_df['Age'], bins)

l = exploration_df.shape[0]

exploration_df = exploration_df.groupby(['Age_bin', 'Survived']).count().reset_index()

exploration_df['PassengerId'] = round(100 * (exploration_df['PassengerId']/l), 2)

exploration_df = exploration_df[['Age_bin', 'Survived', 'PassengerId']]

exploration_df[exploration_df['Survived'] == 1]
import matplotlib.pyplot as plt

import seaborn as sns
exploration_df = train_df

exploration_df['Age_bin'] = pd.cut(train_df['Age'], bins)

sns.countplot('Age_bin', hue='Survived', data=exploration_df)

plt.show()
round(100* pd.crosstab(train_df['Survived'], train_df['Sex'])/train_df.shape[0], 2)
train_df.info()
train_df.drop(columns=['Cabin'], inplace=True)

validation_df.drop(columns=['Cabin'], inplace=True)

train_df.head()
def impute_embarked(sym):

    if sym != sym:

        return 'S'

    return sym

train_df['Embarked'] = train_df['Embarked'].apply(lambda x: impute_embarked(x))

validation_df['Embarked'] = validation_df['Embarked'].apply(lambda x: impute_embarked(x))

train_df.info()
train_df.head()
# train_df.Age.mode()
# mode_age = train_df.Age.mode()[0]



# def impute_age(age):

#     if np.isnan(age):

#         return mode_age

#     return age



# train_df['Age'] = train_df['Age'].apply(lambda x: impute_age(x))

# train_df.info()
from IPython.core.display import display, HTML

HTML('''<script> </script> <form action="javascript:IPython.notebook.execute_cells_above()"><input type="submit" id="toggleButton" value="Run all above Cells"></form>''')
train_df['Sex'] = train_df['Sex'].map({'male':0, 'female':1})

validation_df['Sex'] = validation_df['Sex'].map({'male':0, 'female':1})
train_df.rename(columns={'Sex':'Sex_Male'}, inplace=True)

validation_df.rename(columns={'Sex':'Sex_Male'}, inplace=True)

train_df.head()
embarked = pd.get_dummies(train_df['Embarked'], drop_first=True, prefix='Embarked')

train_df = pd.concat([train_df, embarked], axis=1)

train_df.drop(columns=['Embarked'], inplace=True)



embarked = pd.get_dummies(validation_df['Embarked'], drop_first=True, prefix='Embarked')

validation_df = pd.concat([validation_df, embarked], axis=1)

validation_df.drop(columns=['Embarked'], inplace=True)



train_df.head()
train_df.drop(columns=['Name', 'Ticket', 'PassengerId', 'Age_bin'], inplace=True)

validation_df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)



train_df.head()
Pclass = pd.get_dummies(train_df['Pclass'], drop_first=True, prefix='Pclass')

train_df = pd.concat([train_df, Pclass], axis=1)

train_df.drop(columns=['Pclass'], inplace=True)



Pclass = pd.get_dummies(validation_df['Pclass'], drop_first=True, prefix='Pclass')

validation_df = pd.concat([validation_df, Pclass], axis=1)

validation_df.drop(columns=['Pclass'], inplace=True)



train_df.head()
#! pip install fancyimpute
# Iterative Imputer

# Impute Age

from fancyimpute import IterativeImputer



df_cols = train_df.columns

train_df = pd.DataFrame(IterativeImputer().fit_transform(train_df), columns=df_cols)



df_cols = validation_df.columns

validation_df = pd.DataFrame(IterativeImputer().fit_transform(validation_df), columns=df_cols)
train_df.head()
train_df.isnull().sum()
from IPython.core.display import display, HTML

HTML('''<script> </script> <form action="javascript:IPython.notebook.execute_cells_above()"><input type="submit" id="toggleButton" value="Run all above Cells"></form>''')
# since iterative imputer converts dtypes to float

cols = train_df.columns

cols = cols.drop(['Age', 'Fare'])

for i in cols:

    train_df[i] = pd.to_numeric(train_df[i])

    train_df[i] = train_df[i].astype(int)

    

cols = validation_df.columns

cols = cols.drop(['Age', 'Fare'])

for i in cols:

    validation_df[i] = pd.to_numeric(validation_df[i])

    validation_df[i] = validation_df[i].astype(int)

    

train_df.info()
train_df.head()
sns.boxplot(train_df['Age'], orient='v')

plt.show()
# Lets remove negative age

train_df[train_df['Age'] < 0]
train_df.drop(train_df[train_df['Age'] < 0].index, inplace=True)
sns.boxplot(train_df['Age'], orient='v')

plt.show()
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

sns.distplot(train_df[train_df['Survived'] == 1]['Age'])

plt.title('Survived')

plt.subplot(1, 2, 2)

sns.distplot(train_df[train_df['Survived'] == 0]['Age'])

plt.title('Not Survived')

plt.show()
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

ax = sns.barplot(x='Sex_Male', y='Survived', data=train_df[train_df['Survived'] == 1].groupby('Sex_Male').count().reset_index()[['Sex_Male', 'Survived']], orient='v')

ax.set(ylabel='Count', ylim=(0, 600))

plt.title('Survived')

plt.subplot(1, 2, 2)

ax = sns.barplot(x='Sex_Male', y='Survived', data=train_df[train_df['Survived'] == 0].groupby('Sex_Male').count().reset_index()[['Sex_Male', 'Survived']], orient='v')

ax.set(ylabel='Count', ylim=(0, 600))

plt.title('Not Survived')

plt.show()
ax = sns.barplot(x='Survived', y='Fare', data=train_df.groupby('Survived').mean().reset_index())

ax.set(ylabel='Mean Fare')

plt.show()
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

ax = sns.boxplot(train_df[train_df['Survived'] == 1]['Fare'], orient='v')

ax.set(ylim=(0, 300))

plt.title('Survived')

plt.subplot(1, 2, 2)

ax = sns.boxplot(train_df[train_df['Survived'] == 0]['Fare'], orient='v')

ax.set(ylim=(0, 300))

plt.title('Not Survived')

plt.show()
data = exploration_df.groupby(['Survived', 'Embarked']).count().reset_index()



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

ax = sns.barplot(x = 'Embarked', y = 'PassengerId', data=data[data['Survived']==1], orient='v')

ax.set(ylabel='Count', ylim=(0, 500))

plt.title('Survived')

plt.subplot(1, 2, 2)

ax = sns.barplot(x = 'Embarked', y = 'PassengerId', data=data[data['Survived']==0], orient='v')

ax.set(ylabel='Count', ylim=(0, 500))

plt.title('Not Survived')

plt.show()
data = exploration_df.groupby(['Survived', 'Pclass']).count().reset_index()



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

ax = sns.barplot(x = 'Pclass', y = 'PassengerId', data=data[data['Survived']==1], orient='v')

ax.set(ylabel='Count', ylim=(0, 500))

plt.title('Survived')

plt.subplot(1, 2, 2)

ax = sns.barplot(x = 'Pclass', y = 'PassengerId', data=data[data['Survived']==0], orient='v')

ax.set(ylabel='Count', ylim=(0, 500))

plt.title('Not Survived')

plt.show()
data = exploration_df.groupby(['Survived', 'SibSp']).count().reset_index()



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

ax = sns.barplot(x = 'SibSp', y = 'PassengerId', data=data[data['Survived']==1], orient='v')

ax.set(ylabel='Count', ylim=(0, 500))

plt.title('Survived')

plt.subplot(1, 2, 2)

ax = sns.barplot(x = 'SibSp', y = 'PassengerId', data=data[data['Survived']==0], orient='v')

ax.set(ylabel='Count', ylim=(0, 500))

plt.title('Not Survived')

plt.show()
data = exploration_df.groupby(['Survived', 'Parch']).count().reset_index()



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

ax = sns.barplot(x = 'Parch', y = 'PassengerId', data=data[data['Survived']==1], orient='v')

ax.set(ylabel='Count', ylim=(0, 500))

plt.title('Survived')

plt.subplot(1, 2, 2)

ax = sns.barplot(x = 'Parch', y = 'PassengerId', data=data[data['Survived']==0], orient='v')

ax.set(ylabel='Count', ylim=(0, 500))

plt.title('Not Survived')

plt.show()
# Outlier

sns.boxplot(train_df['Fare'])

plt.show()
train_df.drop(train_df[train_df['Fare'] > 300].index, inplace=True)

sns.boxplot(train_df['Fare'])

plt.show()
from IPython.core.display import display, HTML

HTML('''<script> </script> <form action="javascript:IPython.notebook.execute_cells_above()"><input type="submit" id="toggleButton" value="Run all above Cells"></form>''')
x_columns = train_df.columns.drop('Survived')

X_train = train_df[x_columns]

y_train = train_df['Survived']



print(X_train.shape)

print(y_train.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.7, random_state=1)



print(X_train.shape)

print(y_train.shape)
# Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



X_validation = scaler.transform(validation_df)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)



from sklearn import metrics



print('Accuracy:', metrics.accuracy_score(y_test, y_pred))

print('Sensitivity/Recall:', metrics.recall_score(y_test, y_pred))
#! pip show scikit-learn
# l1 - lasso, l2 - ridge

logreg = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)



print('Accuracy:', metrics.accuracy_score(y_test, y_pred))

print('Sensitivity/Recall:', metrics.recall_score(y_test, y_pred))
# class imbalance

train_df.Survived.value_counts()
logreg = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, class_weight='balanced')

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)



print('Accuracy:', metrics.accuracy_score(y_test, y_pred))

print('Sensitivity/Recall:', metrics.recall_score(y_test, y_pred))
# Grid search for tuning hyperparam - C

from sklearn.model_selection import GridSearchCV

param = {'C': [0.0001, 0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1],

         'penalty': ['l1', 'l2']}



logreg = LogisticRegression(class_weight='balanced', solver='liblinear')

model = GridSearchCV(estimator=logreg,

                     cv=5,

                     param_grid=param,

                     scoring="recall")

model.fit(X_train, y_train)
model.best_score_
model.best_params_
logreg = LogisticRegression(penalty='l1', solver='liblinear', C=0.2, class_weight='balanced')

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)



print('Accuracy:', metrics.accuracy_score(y_test, y_pred))

print('Sensitivity/Recall:', metrics.recall_score(y_test, y_pred))
# ElasticNet - Combines Ridge and Lasso

# Lasso used when - so many features and want to remove some

# Ridge used when - important features and want to reduce coeff power

logreg = LogisticRegression(penalty='elasticnet', solver='saga', C=0.1, class_weight='balanced', l1_ratio=0.9)

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)



print('Accuracy:', metrics.accuracy_score(y_test, y_pred))

print('Sensitivity/Recall:', metrics.recall_score(y_test, y_pred))
from IPython.core.display import display, HTML

HTML('''<script> </script> <form action="javascript:IPython.notebook.execute_cells_above()"><input type="submit" id="toggleButton" value="Run all above Cells"></form>''')
y_valid = logreg.predict(X_validation)
validation_df = pd.read_csv('../input/titanic/test.csv')

submission = pd.DataFrame()

submission = validation_df[['PassengerId']]

submission.head()
submission['Survived'] = y_valid

submission.head()
submission.to_csv('submission.csv',index=False)