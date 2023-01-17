import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


print(os.listdir('../input/'))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

data = [train, test]
def data_info():
    print(train.info())
    print('\n')
    print(test.info())
data_info()
train.head()
train.describe(include='all')
males = train['Sex'] == 'male'
print(round((males.sum() / len(train.index)), 3))
survived = train['Survived'].copy()
passenger_id = test['PassengerId'].copy()

for df in data:
    df.drop('PassengerId', axis=1, inplace=True)
def pivot_var(var_name):
    return train[[var_name, 'Survived']]    \
        .groupby([var_name])                \
        .mean()                             \
        .sort_values(by='Survived',
                     ascending=False)       \
        .round(3)


pivot_var('Pclass')
pivot_var('Sex')
pivot_var('SibSp')
pivot_var('Parch')
pivot_var('Embarked')
sns.set(font_scale=1.5, style='whitegrid')

plot = sns.FacetGrid(train,
                     col='Survived',
                     size=5)

plot.map(plt.hist, 'Age', bins=30);
def age_hist_by(catvar):
    sns.FacetGrid(train,
                  col='Survived',
                  row=catvar,
                  size=3,
                  aspect=1.5)          \
        .map(plt.hist, 'Age', bins=30)


age_hist_by('Pclass')
age_hist_by('Embarked')
sns.FacetGrid(train,
              row='Embarked',
              col='Sex',
              hue='Embarked',
              size=2.5,
              aspect=1.5) \
    .map(sns.barplot,
         'Pclass',
         'Survived',
         ci=68, # using SEM for the error bars
         order=train['Pclass'].unique().sort());
age_hist_by('Sex')
sns.FacetGrid(train,
              col='Survived',
              row='Pclass',
              size=2.5,
              aspect=1.5) \
    .map(plt.hist, 'Fare', bins=15);
colormap = plt.cm.PiYG

sns.set(font_scale=1.4)
plt.figure(figsize=(10, 8))
plt.title('Feature correlation matrix', y=1.02, size=15)

sns.heatmap(train.corr().round(2), square=True,
            linecolor='gray', linewidth=0.1,
            cmap=colormap, annot=True);
train.head()
for df in data:
    df['Family_size'] = df['SibSp'] + df['Parch'] + 1
sns.set(font_scale=1.5, style='whitegrid')
plt.hist(x=train['Family_size'], bins=10, color='C5', edgecolor='black')
plt.axvline(train['Family_size'].mean(), color='black', linestyle='dashed', linewidth=2);
for df in data:
    df['Family_bins'] = pd.cut(df['Family_size'], bins=[0, 1, 3, df['Family_size'].max()])
    print(df['Family_bins'].unique())
for df in data:
    df['Title'] = df['Name'].str.extract(' (\w+)\.', expand=False)
    df['Title'].value_counts()
train['Title'].value_counts()
title_categories = {
    'Capt': 'Other',
    'Col': 'Other',
    'Countess' :'Other',
    'Don': 'Other',
    'Jonkheer': 'Other',
    'Lady': 'Mrs',
    'Major': 'Other',
    'Mlle': 'Other',
    'Mme': 'Other',
    'Ms': 'Miss',
    'Sir': 'Mr'
}
for df in data:
    df['Title'].replace(title_categories, inplace=True)
    print('{}\n'.format(df['Title'].value_counts()))
sns.set(font_scale=1.4)
sns.set_style('whitegrid')
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Title',
            y='Survived',
            data=train,
            ci=68,
            color='C0');
for df in data:
    df['Cabin_count'] = df.groupby(['Cabin'])['Cabin'].transform('count')
for df in data:
    df['Cabin_type'] = pd.cut(df['Cabin_count'].fillna(-1),
                                   bins=[-2, 0, 1, 2, df['Cabin_count'].max()],
                                   labels=['NA', 'Single', 'Double', 'Multiple'])
for df in data:
    print(df['Cabin_type'].unique())
# count of tickets that are the same
for df in data:
    df['Ticket_count'] = df.groupby(['Ticket'])['Ticket'].transform('count')
    df['Ticket_count'] = df['Ticket_count'].astype('category')
for df in data:
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
data_info()
for df in data:
    print('{}\n'.format(df.groupby(['Title', 'Pclass'])['Age'].mean().astype('int')))
for df in data:
    df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))
train.info()
train['Pclass'] = train['Pclass'].astype('category')
test['Pclass'] = test['Pclass'].astype('category')
dummies = ['Pclass', 'Embarked', 'Title', 'Ticket_count', 'Cabin_type', 'Family_bins', 'Sex']
dummy_features = pd.get_dummies(train[dummies])
train = pd.concat([train, dummy_features], axis=1)

dummy_features = pd.get_dummies(test[dummies])
test = pd.concat([test, dummy_features], axis=1)
drop_list = ['Cabin_count', 'Cabin', 'Ticket', 'SibSp', 'Parch', 'Name', 'Family_size']
drop_list.extend(dummies)
train.drop(drop_list, axis=1, inplace=True)
test.drop(drop_list, axis=1, inplace=True)
data_info()
len(train.columns) - len(test.columns)
train.drop([x for x in train.columns if x not in test.columns], axis=1, inplace=True)
test.drop([x for x in test.columns if x not in train.columns], axis=1, inplace=True)
train.info()
RANDOM_SEED = 354135
x_train = train.copy()
y_train = survived.copy()
x_test  = test.copy()
x_train.shape, y_train.shape, x_test.shape
random_forest_parameters = {
    'n_jobs': [-1],
    'random_state': [RANDOM_SEED],
    'n_estimators': [10, 50, 100, 150, 200],
    'max_depth': [4, 8, 12, 16],
    'min_samples_split': [3, 5, 7, 12, 16],
    'min_samples_leaf': [1, 3, 5, 7]
}
forest_cv = GridSearchCV(estimator=RandomForestClassifier(),
                         param_grid=random_forest_parameters,
                         cv=10,
                         verbose=0, # I used 4 on my local notebook, but it seems that on Kaggle the output is very long and makes the reading more difficult, so I set this to zero
                         n_jobs=-1)
forest_cv.fit(x_train, y_train)
print('Best cross validation score: {}'.format(forest_cv.best_score_))
print('Optimal parameters: {}'.format(forest_cv.best_params_))
random_forest = RandomForestClassifier(**forest_cv.best_params_)
random_forest.fit(x_train, y_train)

importances = pd.DataFrame({'score':random_forest.feature_importances_,
                            'feature':x_train.columns})
importances.sort_values('score', ascending=False)
y_pred = forest_cv.predict(x_test)
submission = pd.DataFrame({
        'PassengerId': passenger_id,
        'Survived': y_pred.astype('int')
    })
submission.to_csv('submission.csv', index=False)