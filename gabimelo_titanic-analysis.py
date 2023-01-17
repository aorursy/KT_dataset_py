import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz 
%matplotlib inline

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    cross_val_score, train_test_split, learning_curve, validation_curve
)
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
raw_df = pd.read_csv('../input/train.csv')
df = raw_df.copy()
df.head()
df.info()
print('Amount of rows: {}\n'.format(len(df)))

cols_to_check = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
for col in cols_to_check:
    print('{}:\n{}\n'.format(col, df[col].value_counts()))
    
print('Age range: [{}, {}]\n'.format(np.min(df['Age']), np.max(df['Age'])))

print('Amount of unique names: {}'.format(df['Name'].nunique()))
df['First Name'] = df['Name'].apply(lambda x: x.split('. ')[-1])
df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
df['Last Name'] = df['Name'].apply(lambda x: x.split(',')[0])

print('Unique first names: {}'.format(df['First Name'].nunique()))
print('Unique last names: {}'.format(df['Last Name'].nunique()))
print('Unique titles: {}'.format(df['Title'].nunique()))

assert not df['Title'].isnull().any()

df.head()
df['Title'].value_counts()
plt.figure(figsize=(15,5))
sns.barplot(x=df['Title'], y=df['Age'])
plt.title('Mean age by title')
plt.show()
rows_with_missing_age = df[df['Age'].isnull()]
ax = sns.barplot(x = rows_with_missing_age['Title'].value_counts().index, 
                 y = rows_with_missing_age['Title'].value_counts())
ax.set_ylabel('')
plt.title('Amount of missing values for Age column grouped by Title')
plt.show()
map_means = df.groupby('Title')['Age'].mean().to_dict()

def imput_age(row):
    if np.isnan(row['Age']):
        return map_means[row['Title']]
    else:
        return row['Age']
df['Age'] = df.apply(imput_age, axis=1)

# there should be no more missing values
assert not np.isnan(df['Age']).any()

# means shouldn't have changed:
new_map_means = df.groupby('Title')['Age'].mean().to_dict()
for key, value in map_means.items():
    np.testing.assert_almost_equal(value, new_map_means[key])
df.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket', 'Last Name', 'First Name'], inplace=True)
df.dropna(inplace=True)

df.info()
df['Sex'] = pd.Categorical(df['Sex'])
df['Embarked'] = pd.Categorical(df['Embarked'])
df['Title'] = pd.Categorical(df['Title'])
percentage_survived = df['Survived'].value_counts()[0]/(df['Survived'].value_counts()[0]+df['Survived'].value_counts()[1])
print('{:.2f}% of passengers on the currente data frame died'.format(percentage_survived*100))
plt.figure(figsize=(5,5))
df_corr_title = pd.get_dummies(df.filter(['Title', 'Survived'], axis=1), drop_first=True).corr()
plt.scatter(df_corr_title['Survived'][1:], df_corr_title['Survived'].index[1:])
plt.title('Correlation between titles and survival')
plt.show()
df_corr_wo_title = pd.get_dummies(df.drop('Title', axis=1), drop_first=True).corr()
sns.heatmap(np.abs(df_corr_wo_title), annot=True, fmt='.2f')

# plt.figure(figsize=(5,5))
# df_corr_wo_title = pd.get_dummies(df.drop('Title', axis=1), drop_first=True).corr()
# plt.scatter(df_corr_wo_title['Survived'][1:], df_corr_wo_title['Survived'].index[1:])

plt.title('Correlation between other features')
plt.show()
plt.figure(figsize=(5,7))
df_corr = pd.get_dummies(df, drop_first=True).corr()['Survived']
plt.scatter(df_corr[1:], df_corr.index[1:])
plt.title('Correlation between all features and survival')
plt.show()
sns.barplot(x='Survived', y='Title', data=df)
plt.title('Survival by Title')
plt.show()
sns.barplot(x='Embarked', y='Fare', hue='Pclass', data=df)
plt.title('Fare grouped by embark port and Pclass')
plt.show()
sns.barplot(x='Survived', y='Fare', hue='Pclass', data=df)
plt.title('Fare grouped by survival and Pclass')
plt.show()
g = sns.FacetGrid(df, col='Survived')
g.map(sns.distplot, 'Age')
plt.title('Distribution of Age, separated by Survival')
plt.show()
plt.figure(figsize=(20,5))
ax = sns.barplot(x='Age', y='Survived', data=df)
# ci = None
plt.xticks(rotation=90)
labels = [item.get_text() for item in ax.get_xticklabels()]
ax.set_xticklabels([str(round(float(label), 2)) for label in labels])
plt.title('Survival by Age')
plt.show()
def test_classifier(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cs_score = cross_val_score(classifier, X_train, y_train, cv=10)
    print('***************')
    print(classifier.__class__)
    print('CV Accuracy: {}  +/- {}'.format(cs_score.mean(), cs_score.std()))
    print('Test Set Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
def test_classifiers(X_train, y_train, X_test, y_test):
    classifiers = [DecisionTreeClassifier(random_state=42),
                  LogisticRegression()]
    for classifier in classifiers:
        test_classifier(classifier, X_train, y_train, X_test, y_test)
# this function is from the following notebook:
# https://www.kaggle.com/pmarcelino/data-analysis-and-feature-extraction-with-python/notebook

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation score")
    plt.title(title)
    plt.legend(loc="best")
    return plt
# this function is from the following notebook:
# https://www.kaggle.com/pmarcelino/data-analysis-and-feature-extraction-with-python/notebook

def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name, param_range, cv)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='r', marker='o', markersize=5, label='Training score')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='r')
    plt.plot(param_range, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='Validation score')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='g')
    plt.grid() 
    plt.xscale('log')
    plt.legend(loc='best') 
    plt.xlabel('Parameter') 
    plt.ylabel('Score') 
    plt.ylim(ylim)
    plt.title(title)
# dot_data = tree.export_graphviz(tree_classifier, out_file=None, 
# #                          feature_names=iris.feature_names,  
# #                          class_names=iris.target_names,  
#                          filled=True, rounded=True,  
#                          special_characters=True)  
# graph = graphviz.Source(dot_data)  
# graph
X = pd.get_dummies(df.drop(['Survived'], axis=1), drop_first=True)
y = df['Survived']

X['Family'] = X['SibSp'] + X['Parch']
X.drop(['SibSp', 'Parch'], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
test_classifiers(X_train, y_train, X_test, y_test)
# Plot validation curve
title = 'Validation Curve (Logistic Regression)'
param_name = 'C'
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] 
# could also be written as param_range = np.logspace(-3, 2, num=6)
cv = 10
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
plot_validation_curve(estimator=logreg, title=title, X=X_train, y=y_train, param_name=param_name,
                      ylim=(0.5, 1.01), param_range=param_range)

title = "Learning Curves (Logistic Regression)"
cv = 10
plot_learning_curve(logreg, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=1)
# there are some titles that are equivalent, and some that will be grouped together as 'Other'
titles_dict = {'Capt': 'Other',
               'Col': 'Other',
               'Major': 'Other',
               'Jonkheer': 'Other',
               'Don': 'Other',
               'Sir': 'Other',
               'Dr': 'Other',
               'Rev': 'Other',
               'Countess': 'Other',
               'Mme': 'Mrs',
               'Mlle': 'Miss',
               'Ms': 'Miss',
               'Mr': 'Mr',
               'Mrs': 'Mrs',
               'Miss': 'Miss',
               'Master': 'Master',
               'Lady': 'Other'}

df['Title'] = df['Title'].map(titles_dict)

assert df['Title'].nunique() == 5
assert df['Title'].isnull().any() == False

sns.barplot(x='Title', y='Survived', data=df)
plt.title('Survival by Title')
plt.show()
df['Age in bins'] = pd.Categorical(pd.cut(df['Age'], 
                                   bins=[0, 1, 16, 48, 100], 
                                   labels=['Baby', 'Kid/Teenager','Adult','Elder']))

sns.barplot(x='Age in bins', y='Survived', data=df)
plt.title('Survival by Age bins')
plt.show()
X = pd.get_dummies(df.drop(['Survived'], axis=1), drop_first=True)
X['Family'] = X['SibSp'] + X['Parch']
X.drop(['Age', 'SibSp', 'Parch'], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
test_classifiers(X_train, y_train, X_test, y_test)
# Plot validation curve
title = 'Validation Curve (Logistic Regression)'
param_name = 'C'
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] 
# could also be written as param_range = np.logspace(-3, 2, num=6)
cv = 10
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
plot_validation_curve(estimator=logreg, title=title, X=X_train, y=y_train, param_name=param_name,
                      ylim=(0.5, 1.01), param_range=param_range)

title = "Learning Curves (Logistic Regression)"
cv = 10
plot_learning_curve(logreg, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=1)
df = raw_df.copy()
test_df = pd.read_csv('../input/test.csv')

test_passenger_ids = test_df['PassengerId'].values

test_df.info()
df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df['Name'].str.extract('([A-Za-z]+)\.', expand=False)

map_means = df.groupby('Title')['Age'].mean().to_dict()

def imput_age(row):
    if np.isnan(row['Age']):
        return map_means[row['Title']]
    else:
        return row['Age']
df['Age'] = df.apply(imput_age, axis=1)
test_df['Age'] = test_df.apply(imput_age, axis=1)

df.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'], inplace=True)
df.dropna(inplace=True)

test_df.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'], inplace=True)
test_df = test_df.fillna(test_df.mean()) # 1 missing value on Fare column

df['Sex'] = pd.Categorical(df['Sex'])
df['Embarked'] = pd.Categorical(df['Embarked'])
df['Title'] = pd.Categorical(df['Title'])

test_df['Sex'] = pd.Categorical(test_df['Sex'])
test_df['Embarked'] = pd.Categorical(test_df['Embarked'])
test_df['Title'] = pd.Categorical(test_df['Title'])
X_train = pd.get_dummies(df.drop(['Survived'], axis=1), drop_first=True)
X_test = pd.get_dummies(test_df, drop_first=True)
# make sure both of have the same columns
X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

y_train = df['Survived']
X_train['Family'] = X_train['SibSp'] + X_train['Parch']
X_train.drop(['SibSp', 'Parch'], axis=1, inplace=True)

X_test['Family'] = X_test['SibSp'] + X_test['Parch']
X_test.drop(['SibSp', 'Parch'], axis=1, inplace=True)

classifier = LogisticRegression(random_state=42)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
submission = pd.DataFrame({ 'PassengerId': test_passenger_ids,
                            'Survived': predictions})
submission.to_csv("submission.csv", index=False)