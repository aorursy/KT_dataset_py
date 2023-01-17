from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from collections import Counter

from sklearn.svm import SVC

import seaborn as sns

import pandas as pd

import numpy as np



import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-whitegrid')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_PassengerId = test_data['PassengerId']
train_data.head()
train_data.describe()
train_data.info()
def bar_plot(variable):

    """

    input : variable, example : 'Sex'

    output : bar plot & value count

    """

    # feature

    var = train_data[variable]

    # number of categorical variable

    varValue = var.value_counts()

    

    # visualization

    plt.figure(figsize = (10,4))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel('Frequency')

    plt.title(variable)

    plt.show()

    print('{}: \n{}'.format(variable, varValue))
train_data['Survived'].value_counts().index.values
category1 = ['Survived','Sex','Pclass','Embarked','SibSp','Parch']

for each in category1:

    bar_plot(each)
category2 = ['Cabin','Name','Ticket']

for each in category2:

    print('{}\n'.format(train_data[each].value_counts()))
def hist_plot(variable):

    plt.figure(figsize = (10,4))

    plt.hist(train_data[variable], bins = 55)

    plt.xlabel(variable)

    plt.ylabel('Frequency')

    plt.title('{} distribution with histogram'.format(variable))

    plt.show()
numericVar = ['Fare','Age']

for each in numericVar:

    hist_plot(each)
# Pclass vs Survived

train_data[['Pclass','Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by='Survived', ascending = False)
# Sex vs Survived

train_data[['Sex','Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by='Survived', ascending = False)
# SibSp vs Survived

train_data[['SibSp','Survived']].groupby(['SibSp'], as_index = False).mean().sort_values(by='Survived', ascending = False)
# Parch vs Survived

train_data[['Parch','Survived']].groupby(['Parch'], as_index = False).mean().sort_values(by='Survived', ascending = False)
def detect_outliers(df,features):

    outlier_indices = []

    

    for c in features:

        Q1 = np.percentile(df[c],25) # 1st quartile

        Q3 = np.percentile(df[c],75) # 3rd quartile

        IQR = Q3 - Q1 # IQR

        outlier_step = IQR * 1.5 # Outlier step

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index # detect outlier

        outlier_indices.extend(outlier_list_col) # stack indeces

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    

    return multiple_outliers
train_data.loc[detect_outliers(train_data,['Age','SibSp','Parch','Fare'])]
# Drop outliers

train_data = train_data.drop(detect_outliers(train_data,['Age','SibSp','Parch','Fare']), axis = 0).reset_index(drop = True)
train_data_len = len(train_data)

train_data = pd.concat([train_data, test_data], axis = 0).reset_index(drop = True)
train_data.columns[train_data.isnull().any()]
train_data.isnull().sum()
train_data[train_data['Embarked'].isnull()]
train_data.boxplot(column='Fare', by ='Embarked')

plt.show()
train_data['Embarked'] = train_data['Embarked'].fillna('C')

train_data[train_data['Embarked'].isnull()]
train_data[train_data['Fare'].isnull()]
train_data['Fare'] = train_data['Fare'].fillna(np.mean(train_data[train_data['Pclass'] == 3]['Fare']))

train_data[train_data['Fare'].isnull()]
corr1= ['SibSp', 'Parch', 'Fare', 'Survived',]

fig, ax = plt.subplots(figsize=(14,7))

sns.heatmap(train_data[corr1].corr(), annot = True, fmt = '.2f')

plt.show()

# As we can see, who passengers are paid much money to ticket, their survive rate higher than others
g = sns.factorplot(x = 'SibSp', y = 'Survived', data = train_data, kind = 'bar', size = 7)

g.set_ylabels('Survived Probability')

plt.show()

# Who passengers have much than 2 siblings, their survive rate seems low

# we can make new features for describing these categories
g = sns.factorplot(x = 'Parch', y = 'Survived', data = train_data, kind = 'bar', size = 7)

g.set_ylabels('Survived Probability')

plt.show()

# Passengers watching alone, seem to have low survival rates

# Passengers with more than 3 siblings, seem to have low survival rates

# we can make new features for describing these categories

# we can concat Siblings and Parch
g = sns.factorplot(x = 'Pclass', y = 'Survived', data = train_data, kind = 'bar', size = 7)

g.set_ylabels('Survived Probability')

plt.show()

# passengers watching in high class, survival rates seem high
g = sns.FacetGrid(train_data, col='Survived')

g.map(sns.distplot, 'Age' ,bins=30)

plt.show()

# The graph clearly shows that babies and older people have high survival rates

# passengers over 20 years, couldn't survive

# Passenger distribution is largely between the ages of 15-20

# we could use age feature in train

# we could use age distribution for missing value of age
g = sns.FacetGrid(train_data, col='Survived', row='Pclass', size = 2)

g.map(plt.hist, 'Age' ,bins=30)

g.add_legend()

plt.show()

# pclass is important feature to model training
g = sns.FacetGrid(train_data, row='Embarked', size = 3)

g.map(sns.pointplot, 'Pclass','Survived','Sex' ,bins=30)

g.add_legend()

plt.show()

# women survival rate is higher than men

# Male passengers departing from part c have high survival rate

# we could use embarked and sex features in training
g = sns.FacetGrid(train_data, col='Survived', row='Embarked', size = 3)

g.map(sns.barplot, 'Sex','Fare')

g.add_legend()

plt.show()

# passengers who paid more for ticket have higher survival rates

# we could use fare feature as categorical for training
train_data[train_data['Age'].isnull()]
sns.factorplot(x='Sex', y='Age', data = train_data, kind='box')

plt.show()

# it seems clearly Sex feature is not informative for fill to Age's missing values, because of age distribution looks same
sns.factorplot(x='Sex', y='Age', hue='Pclass', data = train_data, kind='box')

plt.show()

# As average, 1st class passengers are elder than 2nd class, and second class passengers are elder than 3rd class passengers.
sns.factorplot(x='Parch', y='Age', data = train_data, kind='box')

sns.factorplot(x='SibSp', y='Age', data = train_data, kind='box')

plt.show()
train_data['Sex'] = [1 if each == 'male' else 0 for each in train_data['Sex']]
sns.heatmap(train_data[['Age','Sex','SibSp','Parch','Pclass',]].corr(), annot=True)

plt.show()

# Age is not correlation in with Sex, but this is correlation in with Parch,SibSp, Pclass
index_nan_age = list(train_data['Age'][train_data['Age'].isnull()].index)

for each in index_nan_age:

    age_pred = train_data['Age'][((train_data['SibSp'] == train_data.iloc[each]['SibSp']) & (train_data['Parch'] == train_data.iloc[each]['Parch']) & (train_data['Pclass'] == train_data.iloc[each]['Pclass']))].median()

    age_med = train_data['Age'].median()

    if not np.isnan(age_pred):

        train_data['Age'].iloc[each] = age_pred

    else:

        train_data['Age'].iloc[each] = age_med
train_data[train_data['Age'].isnull()]
train_data['Name'].head(10)
name = train_data['Name']

train_data['Title'] = [each.split('.')[0].split(',')[-1].strip() for each in name]
train_data.Title.head(10)
sns.countplot(x='Title', data= train_data)

plt.xticks(rotation = 75)

plt.show()
# convert to categorical

train_data['Title'] = train_data['Title'].replace(['Don','Rev','Dr','Dona','Capt','the Countess','Jonkheer','Col','Sir','Lady','Major',],'Other')

train_data['Title'] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_data["Title"]]

train_data["Title"].head(20)
sns.countplot(x='Title', data= train_data)

plt.xticks(rotation = 75)

plt.show()
g = sns.factorplot(x='Title', y='Survived', data=train_data, kind = 'bar')

g.set_xticklabels(['Master','Mrs','Mr','Other'])

g.set_ylabels("Survival Probability")

plt.show()
train_data.drop(labels = ['Name'], axis = 1, inplace = True)
train_data.head()
train_data = pd.get_dummies(train_data, columns=['Title'])

train_data.head()
train_data['FSize'] = train_data['SibSp'] + train_data['Parch'] + 1
train_data.head()
g = sns.factorplot(x='FSize', y='Survived', data=train_data, kind='bar')

g.set_ylabels("Survival Probability")

plt.show()
train_data['family_size'] = [1 if i < 4.5 else 0 for i in train_data['FSize']]
train_data.head()
sns.countplot(x='family_size', data=train_data)

plt.show()
g = sns.factorplot(x='family_size', y='Survived', data=train_data, kind='bar')

g.set_ylabels("Survival Probability")

plt.show()

# people whose family members are more than 4 , their survival rate is lower

# small families survival rate is higher
train_data = pd.get_dummies(train_data, columns=['family_size'])

train_data.head()
sns.countplot(train_data['Embarked'])

plt.show()
train_data = pd.get_dummies(train_data, columns=['Embarked'])

train_data.head()
tickets = []

for each in list(train_data['Ticket']):

    if not each.isdigit():

        tickets.append(each.replace('.','').replace('/','').strip().split(' ')[0])

    else:

        tickets.append('x')

train_data['Ticket'] = tickets
train_data = pd.get_dummies(train_data, columns=['Ticket'], prefix = 'T')

train_data.head()
sns.countplot(train_data['Pclass'])

plt.show()
train_data['Pclass'] = train_data['Pclass'].astype('category')

train_data = pd.get_dummies(train_data, columns=['Pclass'])

train_data.head()
train_data['Sex'] = train_data['Sex'].astype('category')

train_data = pd.get_dummies(train_data, columns=['Sex'])

train_data.head()
train_data.drop(labels=['PassengerId','Cabin'], axis=1, inplace=True)
train_data.head()
test = train_data[train_data_len:]

test.drop(labels = ['Survived'], axis=1, inplace=True)
train = train_data[:train_data_len]

X_train = train.drop(labels = "Survived", axis = 1)

y_train = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)

print("X_train",len(X_train))

print("X_test",len(X_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))
lr = LogisticRegression()

lr.fit(X_train, y_train)

lr_train_accuracy = round(lr.score(X_train, y_train) * 100,2)

lr_test_accuracy = round(lr.score(X_test, y_test) * 100,2)

print('Train Acc : {}'.format(lr_train_accuracy))

print('Test Acc : {}'.format(lr_test_accuracy))
random_state = 42

classifier = [DecisionTreeClassifier(random_state=random_state),

              SVC(random_state=random_state),

              RandomForestClassifier(random_state=random_state),

              LogisticRegression(random_state=random_state),

              KNeighborsClassifier()]

dt_param_grid = {'min_samples_split' : range(10,500,20),

                 'max_depth': range(1,20,2)}



svc_param_grid = {'kernel' : ['rbf'],

                  'gamma': [0.001, 0.01, 0.1, 1],

                  'C': [1,10,50,100,200,300,100]}



rf_param_grid = {'max_features': [1,3,10],

                 'min_samples_split': [2,3,10],

                 'min_samples_leaf':[1,3,10],

                 'bootstrap':[False],

                 'n_estimators':[100,300],

                 'criterion':['gini']}



lr_param_grid = {'C': np.logspace(-3,3,7),

                 'penalty': ['l1','l2']}



knn_param_grid = {'n_neighbors': np.linspace(1,19,10, dtype=int).tolist(),

                  'weights': ['uniform','distance'],

                  'metric': ['euclidean','manhattan']}



classifier_param = [dt_param_grid,

                    svc_param_grid,

                    rf_param_grid,

                    lr_param_grid,

                    knn_param_grid]
cv_result = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)

    clf.fit(X_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",

             "LogisticRegression",

             "KNeighborsClassifier"]})



g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),

                                        ("rfc",best_estimators[2]),

                                        ("lr",best_estimators[3])],

                                        voting = "soft", n_jobs = -1)

votingC = votingC.fit(X_train, y_train)

print(accuracy_score(votingC.predict(X_test),y_test))
test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)

results = pd.concat([test_PassengerId, test_survived],axis = 1)

results.to_csv("titanic.csv", index = False)