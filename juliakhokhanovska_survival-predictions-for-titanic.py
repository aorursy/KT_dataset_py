import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('whitegrid')
train = pd.read_csv('../input/titanic/train.csv', index_col = 'PassengerId')
test = pd.read_csv('../input/titanic/test.csv', index_col = 'PassengerId')
train.head()
train.info()
test.head()
test.info()
# Removing Cabin with ~80% missing values
train.drop(['Cabin'], axis = 1, inplace = True)
test.drop(['Cabin'], axis = 1, inplace = True)
# Replacing Sex columns with encouded values. 
train.insert(4, 'Male', train['Sex'].str.replace('female', '0').replace('male', '1').astype('category'))
test.insert(3, 'Male', test['Sex'].str.replace('female', '0').replace('male', '1').astype('category'))

# Original columns will be saved in a separate variables
Sex_train = train.pop('Sex')
Sex_test = test.pop('Sex')

# Checking if replacement is correct
print("Distribution in new 'Male' column (train dataset):\n{}".format(train['Male'].value_counts()))
print("\nDistribution in old 'Sex' column (train dataset):\n{}".format(Sex_train.value_counts()))

print("\nDistribution in new 'Male' column (test dataset):\n{}".format(test['Male'].value_counts()))
print("\nDistribution in old 'Sex' column (test dataset):\n{}".format(Sex_test.value_counts()))
# Changing data types to appropriate ones
train['Survived'] = train['Survived'].astype('bool')
train['Pclass'] = train['Pclass'].astype('category')
# Titanic route was Queenstown ---> Cherbourg ---> Southampton ---> New-York, so encouding Ports of Embarkation accordingly to route
train['Embarked'] = train['Embarked'].str.replace('Q', '1').str.replace('C', '2').str.replace('S', '3').astype('float').astype('category')

# Same actions for test set
test['Pclass'] = test['Pclass'].astype('category')
test['Embarked'] = test['Embarked'].str.replace('Q', '1').str.replace('C', '2').str.replace('S', '3').astype('float').astype('category')
# Checking if everything is correct
train.info()
plt.figure(figsize = (8,6))
palette = sns.color_palette(["#e74c3c", "#3498db"])
gender = sns.countplot(train['Male'], palette = palette)
gender.set_xlabel('Gender (Male)')
gender.set_ylabel('Number of passengers')
gender.set_title('Male passengers: {}%\n'.format(int(sum(train['Male'].astype('int'))*100/len(train['Male'].astype('int')))) + 'Female passengers: {}%\n'.format(int(100 - sum(train['Male'].astype('int'))*100/len(train['Male'].astype('int')))))
plt.figure(figsize = (8,6))
palette = sns.color_palette(["#e74c3c", "#3498db"])
survived = sns.countplot(train['Survived'], hue = train['Male'], palette = palette)
survived.set_xlabel('Survived')
survived.set_ylabel('Number of passengers')
survived.set_title('Total survived: {}%'.format(int(sum(train['Survived'])*100/len(train['Survived']))) + 
                   '\nSurvived men: {}%'.format(int(sum(train[train['Male'] == '1']['Survived'])*100/sum(train['Male'].astype('int'))))+
                   '\nSurvived women: {}%'.format(int(sum(train[train['Male'] == '0']['Survived'])*100/(len(train['Male'].astype('int')) - sum(train['Male'].astype('int')))))
                  )
survived.legend(['Women', 'Men'])
plt.figure(figsize = (8,6))
palette = sns.color_palette(["#B6B3B3", "#3498db"])
class_sex_dist = sns.countplot(train['Pclass'], hue = train['Survived'], palette = palette)
class_sex_dist.set_xlabel('Class')
class_sex_dist.set_ylabel('Number of passengers')
class_sex_dist.set_title('1st class passengers: {}%'.format(int(len(train[train['Pclass'] == 1])*100/len(train['Pclass']))) + 
                   '\n2nd class passengers: {}%'.format(int(len(train[train['Pclass'] == 2])*100/len(train['Pclass']))) +
                   '\n3rd class passengers: {}%'.format(int(len(train[train['Pclass'] == 3])*100/len(train['Pclass'])))
                  )
class_sex_dist.legend(["Didn't survive", 'Survived'])
# Age distribution

print('Age distribution in different classes (column "Age")' +
      '\n0-10: {}'.format(len(train[(train['Age'] < 10)])) +
      '\n10-20: {}'.format(len(train[(train['Age'] >= 10) & (train['Age'] < 20)])) +
      '\n20-30: {}'.format(len(train[(train['Age'] >= 20) & (train['Age'] < 30)])) +
      '\n30-40: {}'.format(len(train[(train['Age'] >= 30) & (train['Age'] < 40)])) +
      '\n40-50: {}'.format(len(train[(train['Age'] >= 40) & (train['Age'] < 50)])) +
      '\n50-60: {}'.format(len(train[(train['Age'] >= 50) & (train['Age'] < 60)])) +
      '\n60+: {}'.format(len(train[train['Age'] >= 60])))

sns.distplot(train['Age'], bins = 20)
plt.figure(figsize = (8,6))
age_dist = sns.swarmplot(train['Pclass'], train['Age'], palette = 'PuBuGn', hue = train['Survived'])
age_dist.set_xlabel('Class')
age_dist.set_title('Age distribution in different classes')
age_dist.legend(["Didn't survive", 'Survived'])
# Checking how missing values in age column distributed according class and gender
print('Missing values in Age column per class:\n{}'.format(train[train['Age'].isnull()]['Pclass'].value_counts()))
print('\nMissing values in Age column per gender:\n{}'.format(train[train['Age'].isnull()]['Male'].value_counts()))
# Checking if there are any difference in mean ages for different classes and genders
train.groupby(['Male','Pclass'])['Age'].mean()
# Filling missing values
for i, row in train[train['Age'].isnull()].iterrows():
    train.loc[i, 'Age'] = int(train.groupby(['Male','Pclass'])['Age'].mean()[row['Male']][row['Pclass']])
for i, row in test[test['Age'].isnull()].iterrows():
    test.loc[i, 'Age'] = int(test.groupby(['Male','Pclass'])['Age'].mean()[row['Male']][row['Pclass']])
# Checking, that there are no more missing values in Age column
train[train['Age'].isnull()]
# Here is code to create Age Groups, but I decide to avoid it.

# Creating new columns with age groups
train['AgeGroups'] = train['Age'].apply(lambda x: int(x/10))
test['AgeGroups'] = test['Age'].apply(lambda x: int(x/10))

# Saving old age column separately
Age_train = train.pop('Age')
Age_test = test.pop('Age')

print('Calculated age groups:\n{}'.format(train['AgeGroups'].value_counts().sort_index()))
# Are you curious about the most popular second names? This information is not relevant for our analysis, but can be reviewed out of curiosity.
sns.barplot(train['Name'].apply(lambda second_name: second_name.split(',')[0]).value_counts().head(10).values, train['Name'].apply(lambda second_name: second_name.split(',')[0]).value_counts().head(10).index, palette = 'PuBuGn_r')
# Now let's check how title affect survival
train['Name'].apply(lambda title: title.split(',')[1].split('.')[0]).value_counts()
plt.figure(figsize = (12,6))
palette = sns.color_palette(["#B6B3B3", "#3498db"])
title_plot = sns.countplot(train['Name'].apply(lambda title: title.split(', ')[1].split('.')[0]), order = ['Mr', 'Mrs', 'Miss', 'Master'], palette = palette, hue = train['Survived'])
#title_plot.set_xticklabels(title_plot.get_xticklabels(), rotation=90)
title_plot.set_xlabel('Title')
title_plot.set_ylabel('Number of passengers')
title_plot.legend(loc = 1, labels = ["Didn't survive", 'Survived'])
# It is interesting that all reverends from our dataset passed away. But is that coincidence or part of their outlook on life? Maybe they tried to help others, but suffered themselves. I don't know, so won't make conclusion here.
train[train['Name'].apply(lambda title: title.split(', ')[1].split('.')[0]) == 'Rev']
# If you are interested about the fate of other passengers with honorary titles, then take a look on subset below.
train[train['Name'].apply(lambda title: title.split(', ')[1].split('.')[0]).isin(['Don', 'Dr', 'Ms', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Jonkheer'])]
train['Name'].str.split(', ')[1]#.split('.')[0]
def titles(t):
    t = t.split(', ')[1].split('.')[0]
    if t in ['Mr', 'Don', 'Major', 'Sir', 'Col', 'Capt', 'Jonkheer']:
        return 'Mr'
    elif t in ['Mrs', 'Mme', 'Dona', 'Lady', 'the Countess']:
        return 'Mrs'
    elif t in ['Miss', 'Ms', 'Mlle']:
        return 'Miss'
    else:
        return t
    
train['Title'] = train['Name'].apply(titles).astype('category')
test['Title'] = test['Name'].apply(titles).astype('category')
# Dropping Name column from both datasets
train.drop('Name', axis = 1, inplace = True)
test.drop('Name', axis = 1, inplace = True)
train['Ticket'].nunique() / len(train['Ticket']) * 100
# Dropping Ticket column from both datasets
train.drop('Ticket', axis = 1, inplace = True)
test.drop('Ticket', axis = 1, inplace = True)
# Checking distribution of number of siblings and/or spouses
plt.figure(figsize = (12,6))
palette = sns.color_palette(["#B6B3B3", "#3498db"])
sibsp = sns.countplot(train['SibSp'], hue = train['Survived'], palette = palette)
sibsp.set_xlabel('Number of siblings and/or spouses')
sibsp.set_ylabel('Number of passengers')
sibsp.legend(loc = 1, labels = ["Didn't survive", 'Survived'])
# Checking distribution of number of parents and/or children
plt.figure(figsize = (12,6))
palette = sns.color_palette(["#B6B3B3", "#3498db"])
parch = sns.countplot(train['Parch'], hue = train['Survived'], palette = palette)
parch.set_xlabel('Number of parents and/or children')
parch.set_ylabel('Number of passengers')
parch.legend(loc = 1, labels = ["Didn't survive", 'Survived'])
train['Relatives'] = train['Parch'] + train['SibSp']
Parch_train = train.pop('Parch')
Parch_train = train.pop('SibSp')
test['Relatives'] = test['Parch'] + test['SibSp']
Parch_test = test.pop('Parch')
Parch_test = test.pop('SibSp')
# Checking correlation between Pclass and Fare. 
train['Fare'].corr(train['Pclass'])
plt.figure(figsize = (8,6))
sns.boxplot(train['Pclass'], train['Fare'], palette = 'PuBuGn_r')
train[train['Fare'] > 500]
# Checking missing values in Fare column in test dataset. 
test[test['Fare'].isnull()]
# Filling this cell with mean value for third class that boarded in 3rd route point
test.loc[1044, 'Fare'] = int(test[(test['Pclass'] == 3) & (test['Embarked'] == 3)]['Fare'].mean())
# Checking descriptive statistics for fares
train.groupby(['Pclass'])['Fare'].describe()
# Creating bins
train['FareBins'] = pd.cut(train['Fare'], bins=[0,10,15,30,60,100,550], labels=[1,2,3,4,5,6], include_lowest = True)
test['FareBins'] = pd.cut(test['Fare'], bins=[0,10,15,30,60,100,550], labels=[1,2,3,4,5,6], include_lowest = True)
Fare_train = train.pop('Fare')
Fare_test = test.pop('Fare')
# Checking missing values in Embarked column in train set
train[train['Embarked'].isnull()]
# Looking at distribution depending on class
class_embarked = sns.countplot(train['Pclass'], hue = train['Embarked'], palette = 'PuBuGn')
class_embarked.set_xlabel('Class')
class_embarked.set_ylabel('Number of passengers')
class_embarked.legend(labels = ['Queenstown', 'Cherbourg', 'Southampton'])
train['Embarked'].fillna(3, inplace = True)
train.info()
test.info()
train_dummy = pd.get_dummies(train.drop(['Survived'], axis = 1))
test_dummy = pd.get_dummies(test)
# Uploading prediction libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
train_X = train_dummy
train_y = train['Survived']
test_X = test_dummy
# Train dataset split for model validation
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size = 0.40, random_state = 100)
log_reg = LogisticRegression(max_iter = 500)
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_val)
log_reg.score(X_train, y_train)
print('Confusion matrix:\n{}'.format(confusion_matrix(y_val, log_pred)))
print('\nClassification report:\n{}'.format(classification_report(y_val, log_pred)))
# K-Neighbors Classifier
knn_clf = KNeighborsClassifier()

# Setting parameters for muting
parameters_knn_clf = {'n_neighbors': range(3, 15, 2)}

# Searching for best classificator settings
search_knn_clf = GridSearchCV(knn_clf, parameters_knn_clf, cv = 5)
search_knn_clf.fit(X_train, y_train)
best_knn_clf = search_knn_clf.best_estimator_
print('Best parameters: ', search_knn_clf.best_estimator_)
best_knn_clf.fit(X_train, y_train)
knn_pred = best_knn_clf.predict(X_val)
print('\nPredictions ready')
best_knn_clf.score(X_train, y_train)
print('Confusion matrix:\n{}'.format(confusion_matrix(y_val, knn_pred)))
print('\nClassification report:\n{}'.format(classification_report(y_val, knn_pred)))
# Random Forest Classifier
tree_clf = DecisionTreeClassifier()

# Setting parameters for further search
parameters_tree_clf = {'criterion': ['gini', 'entropy'], 'max_depth': range(5, 36, 5), 'max_leaf_nodes': range(2, 31, 5), 'min_samples_leaf': range(1, 6, 2), 'min_samples_split': range(2,5)}

# Searching for best classificator settings
search_tree_clf = GridSearchCV(tree_clf, parameters_tree_clf, cv = 5)
search_tree_clf.fit(X_train, y_train)
best_tree_clf = search_tree_clf.best_estimator_
print('Best parameters: ', search_tree_clf.best_estimator_)
best_tree_clf.fit(X_train, y_train)
tree_pred = best_tree_clf.predict(X_val)
print('\nPredictions ready')
# t_clf = DecisionTreeClassifier(criterion='entropy', max_depth = 7, max_leaf_nodes = 15, min_samples_leaf = 2, min_samples_split = 2)
# t_clf.fit(X_train, y_train)
# t_pred = t_clf.predict(X_val)
# t_clf.score(X_train, y_train)
print('Confusion matrix:\n{}'.format(confusion_matrix(y_val, tree_pred)))
print('\nClassification report:\n{}'.format(classification_report(y_val, tree_pred)))
# Random Forest Classifier
forest_clf = RandomForestClassifier()

# Setting parameters for further search
parameters_forest_clf = {'n_estimators': range(150, 601, 50), 'max_depth': range(5, 36, 5), 'max_leaf_nodes': range(5, 31, 5), 'min_samples_leaf': range(1, 6, 2), 'min_samples_split': range(2,5)}

# Searching for best classificator settings
search_forest_clf = RandomizedSearchCV(forest_clf, parameters_forest_clf, cv = 5)
search_forest_clf.fit(X_train, y_train)
best_forest_clf = search_forest_clf.best_estimator_
print('Best parameters: ', search_forest_clf.best_estimator_)
best_forest_clf.fit(X_train, y_train)
forest_pred = best_forest_clf.predict(X_val)
print('\nPredictions ready')
best_forest_clf.score(X_train, y_train)
print('Confusion matrix:\n{}'.format(confusion_matrix(y_val, forest_pred)))
print('\nClassification report:\n{}'.format(classification_report(y_val, forest_pred)))
booster_clf = GradientBoostingClassifier()

# Setting parameters for muting
parameters_booster_clf = {'n_estimators': range(50, 301, 50)}

# Searching for best classificator settings
search_booster_clf = GridSearchCV(booster_clf, parameters_booster_clf, cv = 5)
search_booster_clf.fit(X_train, y_train)
best_booster_clf = search_booster_clf.best_estimator_
print('Best parameters: ', search_booster_clf.best_estimator_)
best_booster_clf.fit(X_train, y_train)
booster_pred = best_booster_clf.predict(X_val)
print('\nPredictions ready')
best_booster_clf.score(X_train, y_train)
print('Confusion matrix:\n{}'.format(confusion_matrix(y_val, booster_pred)))
print('\nClassification report:\n{}'.format(classification_report(y_val, booster_pred)))
svc_clf = SVC()

# Setting parameters for further search
parameters_svc_clf = {'gamma': range(0, 100, 1)}

# Searching for best classificator settings
search_svc_clf = GridSearchCV(svc_clf, parameters_svc_clf, cv = 5)
search_svc_clf.fit(X_train, y_train)
best_svc_clf = search_booster_clf.best_estimator_
print('Best parameters: ', search_svc_clf.best_estimator_)
best_svc_clf.fit(X_train, y_train)
svc_pred = best_svc_clf.predict(X_val)
print('\nPredictions ready')
best_svc_clf.score(X_train, y_train)
print('Confusion matrix:\n{}'.format(confusion_matrix(y_val, svc_pred)))
print('\nClassification report:\n{}'.format(classification_report(y_val, svc_pred)))
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)#, eval_metric='auc', verbose=True)
xgb_pred = xgb_clf.predict(X_val)
print('Predictions ready')
xgb_clf.score(X_train, y_train)
print('Confusion matrix:\n{}'.format(confusion_matrix(y_val, xgb_pred)))
print('\nClassification report:\n{}'.format(classification_report(y_val, xgb_pred)))
# Predicting results
predictions = best_tree_clf.predict(test_dummy)
# Forming dataframe
submission = pd.DataFrame({'PassengerId': test.index, 'Survived': predictions.astype('int')})
submission.head()
# Let's take a look on distribution
palette = sns.color_palette(["#B6B3B3", "#3498db"])
title_plot = sns.countplot(submission['Survived'], palette = palette)
title_plot.set_ylabel('Number of passengers')
submission.to_csv('/kaggle/working/submission.csv', index = False)