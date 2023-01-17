# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image

# data analysis / manipulation
import numpy as np
import pandas as pd
import re

# statistics
from scipy import stats

# machine learning
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
test_df = pd.read_csv("../input/titanic/test.csv")
train_df = pd.read_csv("../input/titanic/train.csv")
combined_df = pd.concat([train_df, test_df])
train_df.head()
train_df.describe(include = 'all')
train_df.isna().sum()
train_df.Name.unique().size
ax = sns.barplot(x = train_df.Sex.value_counts().index, y = train_df.Sex.value_counts().values)
ax.set(xlabel = 'Gender', ylabel = '# of Passengers per Gender', title = '# of Passengers per Gender')
ax.set_xticklabels(['Male','Female'])
ax = sns.barplot(x = train_df.Survived.value_counts().index, y = train_df.Survived.value_counts().values)
ax.set(xlabel = 'Survival', ylabel = '# of Passengers', title = 'Passenger Surviability from the Training Set')
ax.set_xticklabels(['Did Not Survive','Survived'])
ax = sns.countplot(x = 'Survived', hue = 'Sex', data = train_df)
ax.set(xlabel = 'Survival', ylabel = '# of Passengers', title = 'Passenger Surviability from the Training Set')
ax.set_xticklabels(['Did Not Survive','Survived'])
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1 
sns.countplot(train_df['FamilySize'])
ax = sns.distplot(train_df.Age)
ax.set(xlabel = 'Age', ylabel = '', title = 'Distribution of Passengers by Age')
ax = sns.distplot(train_df.Fare)
ax.set(xlabel = 'Fare (Shillings/Pounds)', ylabel = 'Dist', title = 'Distribution of Passengers Fares')
(train_df.Fare.max()/91.1)*(91.1)
(train_df.Fare.max()/91.1)*(126.06)
adjusted = 708.94/9.7
adjusted
train_df['Fare_2020'] = train_df.Fare * adjusted
ax = sns.distplot(train_df.Fare_2020)
ax.set(xlabel = 'Fare (£)', ylabel = 'Dist', title = 'Distribution of Passengers Fare (2020 Prices)')
train_df.loc[train_df.Fare_2020.idxmax()]
facet = sns.FacetGrid(train_df, hue = 'Survived', height = 10)
facet.map(sns.distplot, 'Age', bins = 20)
facet.set(xlim = (0, train_df['Age'].max()))
facet.add_legend()
sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = train_df, split = True)
ax = sns.barplot(x = train_df.Embarked.value_counts().index, y = train_df.Embarked.value_counts().values)
ax.set(xlabel = 'Point of Embarkment', ylabel = '# of Passengers', title = '# of Passengers Embarked at Location Specified')
ax.set_xticklabels(['Southampton','Cherbourg','Queenstown'])
fg = sns.FacetGrid(train_df, row = 'Embarked')
fg.map(sns.pointplot, 'Pclass','Survived','Sex')
fg.add_legend()
gender_age = train_df[['Age','Sex','Survived']]
gender_age.dropna(inplace = True)
gender_dummies = pd.get_dummies(gender_age['Sex'])
gender_dummies
gender_age_dummies = gender_age.join(gender_dummies)
gender_age_dummies
survival_gender = gender_age_dummies.groupby('Survived')[['female','male']].mean()
survival_gender
sns.heatmap(survival_gender, annot = True, cmap = "YlGnBu")
combined_df['Titles'] = combined_df.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
min_count = 10
title_names = (combined_df['Titles'].value_counts() < min_count)

combined_df['Titles'].value_counts()
combined_df['Titles'] = combined_df['Titles'].apply(lambda x: 'Other' if title_names.loc[x] == True else x)
combined_df['Titles'].value_counts()
sns.factorplot('Titles', 'Survived', data = combined_df)
# combined_df.drop(columns = 'PassengerId', inplace = True)
sns.heatmap(combined_df.corr(), annot = True)
combined_df['Age'] = combined_df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))
combined_df.isna().sum()
combined_df['Fare'].fillna(combined_df['Fare'].mode()[0], inplace = True)
combined_df.isna().sum()
print(train_df.info())
print("----------------")
print(test_df.info())
combined_df.head()
title_category = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Other':5}

combined_df['Titles'] = combined_df['Titles'].map(title_category)

combined_df.head()
gender_category = {"male": 0, "female": 1}

combined_df['Sex'] = combined_df['Sex'].map(gender_category)

combined_df.head()
embarkment_category = {"S": 0, "C": 1, "Q": 2}

combined_df['Embarked'] = combined_df['Embarked'].map(embarkment_category)

combined_df.head()
combined_df['Embarked'].fillna(method = 'bfill', inplace = True)
combined_df.isna().sum()
combined_df['FamilySize'] = combined_df['SibSp'] + combined_df['Parch'] + 1
combined_df = combined_df.drop(columns = ['SibSp','Parch','Ticket','Name','Cabin'])
combined_df['Fare'] = (combined_df['Fare'] - combined_df['Fare'].mean())/(combined_df['Fare'].max() - combined_df['Fare'].min())
combined_df['Age'] = (combined_df['Age'] - combined_df['Age'].mean())/(combined_df['Age'].max() - combined_df['Age'].min())
test = combined_df[combined_df['Survived'].isna()].drop(['Survived'], axis = 1)
test.head()
train = combined_df[combined_df['Survived'].notna()]
train['Survived'] = train['Survived'].astype(np.int8)
train.isna().sum()
test.shape
X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived', 'PassengerId'], axis = 1), train['Survived'], test_size = 0.22, random_state = 42)
X_train.shape, y_train.shape, X_test.shape
logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
logistic_acc = round(accuracy_score(y_pred, y_test)*100, 2)
print(logistic_acc)
correl = pd.DataFrame(combined_df.columns.delete(0))
correl.columns = ['Features']
correl["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
correl = correl.sort_values(by = 'Coefficient Estimate', ascending = False)
correl = correl.astype({'Features': str, 'Coefficient Estimate': float})
g = sns.pointplot(x = correl.Features, y = correl['Coefficient Estimate'], join = False)
g.set_xticklabels(labels = correl.Features, rotation = 45)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
decision_tree_acc = round(decision_tree.score(X_train, y_train)* 100, 2)
decision_tree_acc
random_forest = RandomForestClassifier()

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

random_forest_accuracy = round(random_forest.score(X_train, y_train)*100,2)

print(random_forest_accuracy)
guassian = GaussianNB()
guassian.fit(X_train, y_train)
y_pred = guassian.predict(X_test)
guassian_acc = round(accuracy_score(y_pred, y_test) * 100, 2)
print(guassian_acc)
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
svc_acc = round(accuracy_score(y_pred, y_test) * 100, 2)
print(svc_acc)
linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
y_pred = linear_svc.predict(X_test)
linear_svc_acc = round(accuracy_score(y_pred, y_test) * 100, 2)
print(linear_svc_acc)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn_acc = round(accuracy_score(y_pred, y_test) * 100, 2)
print(knn_acc)
submission = test['PassengerId']
prediction = random_forest.predict(test.drop('PassengerId', axis = 1))

submission_csv = pd.DataFrame({'PassengerId': submission, 'Survived': prediction})
submission_csv.to_csv('submission.csv', index = False)