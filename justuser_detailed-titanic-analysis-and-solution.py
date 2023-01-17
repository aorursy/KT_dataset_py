import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
filePath = "../input/train.csv"
train = pd.read_csv(filePath)
filePath = "../input/test.csv"
test = pd.read_csv(filePath)
train.head()
plt.figure(figsize=(14, 12))

# don't forget to set titles
plt.subplot(211)
sns.heatmap(train.isnull(), yticklabels=False)
plt.subplot(212)
sns.heatmap(test.isnull(), yticklabels=False)
plt.figure(figsize=(14, 12))

plt.subplot(321)
sns.countplot('Survived', data=train)
plt.subplot(322)
sns.countplot('Sex', data=train, hue='Survived')
plt.subplot(323)
sns.distplot(train['Age'].dropna(), bins=25)
plt.subplot(324)
sns.countplot('Pclass', data=train, hue='Survived')
plt.subplot(325)
sns.countplot('SibSp', data=train)
plt.subplot(326)
sns.countplot('Parch', data=train)
plt.figure(figsize=(10,8))
sns.heatmap(train.isnull(), yticklabels=False)
train.head()
def get_title(pasngr_name):
    
    index_1 = pasngr_name.find(', ') + 2
    index_2 = pasngr_name.find('. ') + 1
    
    return pasngr_name[index_1:index_2]
train['Title'] = train['Name'].apply(get_title)
test['Title'] = test['Name'].apply(get_title)
plt.figure(figsize=(16, 10))
sns.boxplot('Title', 'Age', data=train)
train.Title.unique()
age_by_title = train.groupby('Title')['Age'].mean()
print(age_by_title)
def fill_missing_ages(cols):
    age = cols[0]
    titles = cols[1]
    
    if pd.isnull(age):
        return age_by_title[titles]
    else:
        return age
train['Age'] = train[['Age', 'Title']].apply(fill_missing_ages, axis=1)
test['Age'] = test[['Age', 'Title']].apply(fill_missing_ages, axis=1)

#and one Fare value in the test set
test['Fare'].fillna(test['Fare'].mean(), inplace = True)

plt.figure(figsize=(14, 12))

plt.subplot(211)
sns.heatmap(train.isnull(), yticklabels=False)
plt.subplot(212)
sns.heatmap(test.isnull(), yticklabels=False)
sns.countplot('Embarked', data=train)
train['Embarked'].fillna('S', inplace=True)
sns.heatmap(train.isnull(), yticklabels=False)
plt.figure(figsize=(10,8))
sns.heatmap(train.drop('PassengerId', axis=1).corr(), annot=True)
plt.figure(figsize=(10, 8))
sns.violinplot('Survived', 'Age', data=train)
plt.figure(figsize=(10, 8))
sns.violinplot('Sex', 'Age', data=train, hue='Survived', split=True)
grid = sns.FacetGrid(train, col='Pclass', hue="Survived", size=4)
grid = grid.map(sns.swarmplot, 'Sex', 'Age', order=["female"])
plt.figure(figsize=(10, 8))
sns.countplot('Pclass', data=train, hue='Survived')
plt.figure(figsize=(14, 6))

plt.subplot(121)
sns.barplot('Pclass', 'Fare', data=train)
plt.subplot(122)
sns.barplot('Pclass', 'Age', data=train)
sns.lmplot('Age', 'Fare', data=train, hue='Pclass', fit_reg=False, size=7)
train["FamilySize"] = train["SibSp"] + train["Parch"]
test["FamilySize"] = test["SibSp"] + test["Parch"]
train.head()
plt.figure(figsize=(14, 6))

plt.subplot(121)
sns.barplot('FamilySize', 'Survived', data=train)
plt.subplot(122)
sns.countplot('FamilySize', data=train, hue='Survived')
grid = sns.FacetGrid(train, col='Sex', size=6)
grid = grid.map(sns.barplot, 'FamilySize', 'Survived')
plt.figure(figsize=(10, 8))
sns.countplot('FamilySize',  data=train, hue='Pclass')
sns.countplot('Embarked', data=train, hue='Survived')
sns.countplot('Embarked', data=train, hue='Pclass')
def has_cabin(pasngr_cabin):
    
    if pd.isnull(pasngr_cabin):
        return 0
    else:
        return 1
    
train['CabinKnown'] = train['Cabin'].apply(has_cabin)
test['CabinKnown'] = test['Cabin'].apply(has_cabin)
sns.countplot('CabinKnown', data=train, hue='Survived')
def get_age_categories(age):
    if(age <= 16):
        return 'child'
    elif(age > 16 and age <= 50):
        return 'adult'
    else:
        return 'elder'
    
train['AgeCategory'] = train['Age'].apply(get_age_categories)
test['AgeCategory'] = test['Age'].apply(get_age_categories)
sns.countplot('AgeCategory', data=train, hue='Survived')
def get_family_category(family_size):
    
    if(family_size > 3):
        return 'WithLargeFamily'
    elif(family_size > 0 and family_size<= 3):
        return 'WithFamily'
    else:
        return 'TraveledAlone'
    
train['FamilyCategory'] = train['FamilySize'].apply(get_family_category)
test['FamilyCategory'] = test['FamilySize'].apply(get_family_category)
print(train.Title.unique())
plt.figure(figsize=(12, 10))
sns.countplot('Title', data=train)
titles_to_cats = {
    'HighClass': ['Lady.', 'Sir.'],
    'MiddleClass': ['Mr.', 'Mrs.'],
    'LowClass': []
}
plt.figure(figsize=(10, 8))
sns.distplot(train['Fare'])
train['Sex'] = train['Sex'].astype('category').cat.codes
test['Sex'] = test['Sex'].astype('category').cat.codes
train[['Name', 'Sex']].head()
embarkedCat = pd.get_dummies(train['Embarked'])
train = pd.concat([train, embarkedCat], axis=1)
train.drop('Embarked', axis=1, inplace=True)

embarkedCat = pd.get_dummies(test['Embarked'])
test = pd.concat([test, embarkedCat], axis=1)
test.drop('Embarked', axis=1, inplace=True)

train[['Q', 'S', 'C']].head()
# for the train set
familyCat = pd.get_dummies(train['FamilyCategory'])
train = pd.concat([train, familyCat], axis=1)
train.drop('FamilyCategory', axis=1, inplace=True)

ageCat = pd.get_dummies(train['AgeCategory'])
train = pd.concat([train, ageCat], axis=1)
train.drop('AgeCategory', axis=1, inplace=True)

#and for the test
familyCat = pd.get_dummies(test['FamilyCategory'])
test = pd.concat([test, familyCat], axis=1)
test.drop('FamilyCategory', axis=1, inplace=True)

ageCat = pd.get_dummies(test['AgeCategory'])
test = pd.concat([test, ageCat], axis=1)
test.drop('AgeCategory', axis=1, inplace=True)
plt.figure(figsize=(14,12))
sns.heatmap(train.drop('PassengerId', axis=1).corr(), annot=True)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

classifiers = [
    RandomForestClassifier(),
    KNeighborsClassifier(),
    SVC(),
    DecisionTreeClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    ExtraTreesClassifier(),
    LogisticRegression()
]
X_train = train.drop(['PassengerId', 'Survived', 'SibSp', 'Parch', 'Ticket', 'Name', 'Cabin', 'Title', 'FamilySize'], axis=1)
y_train = train['Survived']

X_final = test.drop(['PassengerId', 'SibSp', 'Parch', 'Ticket', 'Name', 'Cabin', 'Title', 'FamilySize'], axis=1)
from sklearn.model_selection import KFold

# n_splits=5
cv_kfold = KFold(n_splits=10)
from sklearn.model_selection import cross_val_score

class_scores = []
for classifier in classifiers:
    class_scores.append(cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=cv_kfold))
    
class_mean_scores = []
for score in class_scores:
    class_mean_scores.append(score.mean())
scores_df = pd.DataFrame({
    'Classifier':['Random Forest', 'KNeighbors', 'SVC', 'DecisionTreeClassifier', 'AdaBoostClassifier', 
                  'GradientBoostingClassifier', 'ExtraTreesClassifier', 'LogisticRegression'], 
    'Scores': class_mean_scores
})

print(scores_df)
sns.factorplot('Scores', 'Classifier', data=scores_df, size=6)
g_boost = GradientBoostingClassifier()
g_boost.get_params().keys()
from sklearn.model_selection import GridSearchCV

param_grid = {
    'loss': ['deviance', 'exponential'],
    'min_samples_leaf': [2, 5, 10],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [100],
    'max_depth': [3, 5, 10, 20]
}

grid_cv = GridSearchCV(g_boost, param_grid, scoring='accuracy', cv=cv_kfold)
grid_cv.fit(X_train, y_train)
grid_cv.best_estimator_
print(grid_cv.best_score_)
print(grid_cv.best_params_)
g_boost = GradientBoostingClassifier(min_samples_split=5, loss='deviance', n_estimators=1000, 
                                     max_depth=3, min_samples_leaf=2)
g_boost.fit(X_train, y_train)
feature_values = pd.DataFrame({
    'Feature': X_final.columns,
    'Importance': g_boost.feature_importances_
})

print(feature_values)
sns.factorplot('Importance', 'Feature', data=feature_values, size=6)
prediction = g_boost.predict(X_final)
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': prediction
})
#submission.to_csv('submission.csv', index=False)