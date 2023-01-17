import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(color_codes=True)

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ( ExtraTreesClassifier, AdaBoostClassifier, 
                            GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier )

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import random
train = pd.read_csv( '../input/train.csv' )
test = pd.read_csv('../input/test.csv')

print("Train Data")
print("Number of Rows : ", train.shape[0])
print("Number of Columns : ", train.shape[1])

print("\nTest Data")
print("Number of Rows : ", test.shape[0])
print("Number of Columns : ", test.shape[1])
train.head()
train.dtypes 
train.describe()
train.isnull().sum()
test.isnull().sum()
plt.figure(figsize=(7,7))
plt.ylabel("Number of People Who Survived")
ax = sns.countplot( x = 'Survived', data = train )
ax.set_xticklabels(['Not Survived','Survived'])
ax.set_xlabel('Target class')
ax.set_title("Distribution of target class")
ax.set_ylabel("Count")
plt.figure(2, figsize=(20,20))
the_grid = plt.GridSpec(2, 2)

plt.subplot(the_grid[0, 0])

ax = sns.countplot( x = 'Pclass', data = train )
ax.set_xticklabels(['Upper','Middle', 'Lower'])
ax.set_xlabel('Passenger Class', fontsize=16, labelpad=10)
ax.set_title("Distribution of passenger class", fontsize=18)
ax.set_ylabel("Count", fontsize=16)
ax.tick_params(labelsize=14)

plt.subplot(the_grid[0, 1])
ax = sns.countplot( x = "Pclass", hue = "Survived", data = train )
ax.set_xticklabels(['Upper','Middle', 'Lower'])
ax.set_xlabel('Passenger Class',fontsize=16, labelpad=10)
ax.set_title("Distribution of pclass w.r.t target class", fontsize=18)
ax.set_ylabel("Count", fontsize=16)
ax.tick_params(labelsize=14)

plt.suptitle("Passenger Class", x=0.5, y=0.925, fontsize=20)
plt.figure(figsize=(7,7))
ax = sns.barplot( x = "Pclass", y = "Survived", data = train )
ax.set_xticklabels(['Upper','Middle', 'Lower'])
ax.set_xlabel('Passenger Class')
ax.set_title("Distribution of pclass w.r.t survival rate")
ax.set_ylabel("Survived")
temp_age = train.Age
temp_age.dropna(inplace=True)

plt.figure(2, figsize=(20,20))
the_grid = plt.GridSpec(2, 1)

plt.subplot(the_grid[0, 0])

ax = sns.distplot( temp_age, kde = False, bins = 80, color = 'r', hist_kws=dict(edgecolor="black", linewidth = 1) )
ax.set_xticks( range(0, 90, 5) )
ax.set_xlabel('Age', fontsize=16, labelpad=10)
ax.set_title("Distribution of age", fontsize=18)
ax.set_ylabel("Count", fontsize=16)
ax.tick_params(labelsize=14)

plt.subplot(the_grid[1, 0])
ax = sns.barplot( x = "Age", y = "Survived", data = train, ci=None )
ax.set_xticks( range(0, 90, 5) )
ax.set_xticklabels(range(0, 90, 5))
ax.set_xlabel('Age',fontsize=16, labelpad=10)
ax.set_title("Distribution of age w.r.t survival rate", fontsize=18)
ax.set_ylabel("Survived", fontsize=16)
ax.tick_params(labelsize=14)

plt.suptitle("Age", x=0.5, y=0.925, fontsize=20)
plt.figure(2, figsize=(20,20))
the_grid = plt.GridSpec(2, 2)

plt.subplot(the_grid[0, 0])

ax = sns.countplot( x = "Sex", data = train )
ax.set_xticklabels(['Male','Female'])
ax.set_xlabel('Sex', fontsize=16, labelpad=10)
ax.set_title("Distribution of Sex", fontsize=18)
ax.set_ylabel("Count", fontsize=16)
ax.tick_params(labelsize=14)

plt.subplot(the_grid[0, 1])
ax = sns.barplot( x="Sex", y="Survived", data=train, ci=None )
ax.set_xticklabels(['Male','Female'])
ax.set_xlabel('Sex',fontsize=16, labelpad=10)
ax.set_title("Distribution of sex w.r.t survival rate", fontsize=18)
ax.set_ylabel("Survived", fontsize=16)
ax.tick_params(labelsize=14)

plt.suptitle("Sex", x=0.5, y=0.925, fontsize=20)
plt.figure(figsize=(8,6))
ax = sns.pointplot( x = "Pclass", y = "Survived", hue = "Sex", data = train, palette={"male": "g", "female": "m"}, markers=["^", "o"], linestyles=["-", "--"] )
ax.set_xticklabels(['Upper','Middle', 'Lower'])
ax.set_xlabel('Passenger class')
ax.set_title("Distribution of sex and passenger class w.r.t survival rate")
ax.set_ylabel("Survived")
plt.figure(2, figsize=(20,20))
the_grid = plt.GridSpec(2, 2)

plt.subplot(the_grid[0, 0])

ax = sns.countplot( train['SibSp'] )
ax.set_xlabel('Siblings/Spouses', fontsize=16, labelpad=10)
ax.set_title("Number of siblings/spouses on ship", fontsize=18)
ax.set_ylabel("Count", fontsize=16)
ax.tick_params(labelsize=14)

plt.subplot(the_grid[0, 1])
ax = sns.pointplot( x = "SibSp", y = "Survived", data = train, markers=["o"], linestyles=["--"], color="#bb3f3f", ci=None )
ax.set_xlabel('Siblings/Spouses',fontsize=16, labelpad=10)
ax.set_title("Survival rate of siblings/spouses on ship", fontsize=18)
ax.set_ylabel("Survived", fontsize=16)
ax.tick_params(labelsize=14)

plt.suptitle("Siblings/Spouse", x=0.5, y=0.925, fontsize=20)
plt.figure(2, figsize=(20,20))
the_grid = plt.GridSpec(2, 2)

plt.subplot(the_grid[0, 0])

ax = sns.countplot( train['Parch'] )
ax.set_xlabel('Parents/Children', fontsize=16, labelpad=10)
ax.set_title("Number of parents/children on ship", fontsize=18)
ax.set_ylabel("Count", fontsize=16)
ax.tick_params(labelsize=14)

plt.subplot(the_grid[0, 1])
ax = sns.pointplot( x = "Parch", y = "Survived", data = train, markers=["^"], linestyles=["--"], color="#6699ff", ci=None )
ax.set_xlabel('Parents/Children',fontsize=16, labelpad=10)
ax.set_title("Survival rate of parents/children on ship", fontsize=18)
ax.set_ylabel("Survived", fontsize=16)
ax.tick_params(labelsize=14)

plt.suptitle("Parents/Children", x=0.5, y=0.925, fontsize=20)
plt.figure(2, figsize=(20,20))
the_grid = plt.GridSpec(2, 1)

plt.subplot(the_grid[0, 0])

ax = sns.distplot( train['Fare'], kde = False, bins = 400, color = '#003399', hist_kws=dict(edgecolor= 'black', linewidth=1) )
ax.set_xticks( range(0, 150, 5) )
ax.set_xlim(0,150)
ax.set_xlabel('Fare', fontsize=16, labelpad=10)
ax.set_title("Distribution of fare", fontsize=18)
ax.set_ylabel("Count", fontsize=16)
ax.tick_params(labelsize=14)

plt.subplot(the_grid[1, 0])
ax = sns.barplot( x = "Fare", y = "Survived", data = train, ci=None )
ax.set_xlim(0,150)
ax.set_xticks( range(0, 150, 5) )
ax.set_xticklabels(range(0, 150, 5))
ax.set_xlabel('Fare',fontsize=16, labelpad=10)
ax.set_title("Distribution of fare w.r.t survival rate", fontsize=18)
ax.set_ylabel("Survived", fontsize=16)
ax.tick_params(labelsize=14)

plt.suptitle("Fare", x=0.5, y=0.925, fontsize=20)
plt.figure(2, figsize=(20,20))
the_grid = plt.GridSpec(2, 2)

plt.subplot(the_grid[0, 0])
ax = sns.countplot( train['Embarked'] )
ax.set_xticklabels(['Southampton', 'Cherbourg', 'Queenstown'])
ax.set_xlabel('Port Of Embarkation', fontsize=16, labelpad=10)
ax.set_title("Distribution of port of embarkation", fontsize=18)
ax.set_ylabel("Count", fontsize=16)
ax.tick_params(labelsize=14)

plt.subplot(the_grid[0, 1])
ax = sns.barplot( x = "Embarked", y = "Survived", data = train, ci=None )
ax.set_xticklabels(['Southampton', 'Cherbourg', 'Queenstown'])
ax.set_xlabel('Port Of Embarkation',fontsize=16, labelpad=10)
ax.set_title("Distribution of port of embarkation w.r.t survival rate", fontsize=18)
ax.set_ylabel("Survived", fontsize=16)
ax.tick_params(labelsize=14)

plt.suptitle("Port of Embarkation", x=0.5, y=0.925, fontsize=20)
train.Ticket.value_counts()
train.Cabin.isnull().sum()
train.Cabin.value_counts()
plt.figure(figsize = (15,12)) 
ax = sns.heatmap( train.corr(), annot=True, cmap = sns.diverging_palette(250, 10, as_cmap=True) )
ax.set_title("Heat map showing correlation between attributes")
# For train data....
# Make a new column with name "Titles" and assign every value 0
train['Titles'] = 0

# Loop over each name, extract title using regular expression and assign to new column "Title"
for i in train:
    train['Titles'] = train.Name.str.extract('([A-Za-z]+)\.', expand=False)   
    
train['Titles'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

# For test data....
# Make a new column with name "Titles" and assign every value 0
test['Titles'] = 0

# Loop over each name, extract title using regular expression and assign to new column "Title"
for i in test:
    test['Titles'] = test.Name.str.extract('([A-Za-z]+)\.', expand=False)   
    
test['Titles'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

plt.figure(2, figsize=(20,20))
the_grid = plt.GridSpec(2, 2)

plt.subplot(the_grid[0, 0])
ax = sns.countplot(x='Titles', data=train)
ax.set_xlabel('Titles', fontsize=16, labelpad=10)
ax.set_title("Distribution of titles", fontsize=18)
ax.set_ylabel("Count", fontsize=16)
ax.tick_params(labelsize=14)

plt.subplot(the_grid[0, 1])
ax = sns.barplot(x='Titles', y='Survived', data=train, ci=None)
ax.set_xlabel('Titles', fontsize=16, labelpad=10)
ax.set_title("Distribution of titles w.r.t survival rate", fontsize=18)
ax.set_ylabel("Survived", fontsize=16)
ax.tick_params(labelsize=14)

plt.suptitle("Titles", x=0.5, y=0.925, fontsize=20)
# For train data....
# A new column family size is created by adding # of parents/children and # of siblings/spouses
train['FamilySize'] = train['Parch'] + train['SibSp'] + 1

# Create a column named Alone with default values of 0
train['Alone'] = 0

# If family size == 1, then the person is alone. Thus, place 1 in column 'Alone'
train.loc[ train.FamilySize == 1, 'Alone' ] = 1

# For test data....
# A new column family size is created by adding # of parents/children and # of siblings/spouses
test['FamilySize'] = test['Parch'] + test['SibSp'] + 1

# Create a column named Alone with default values of 0
test['Alone'] = 0

#if family size == 1, then the person is alone. Thus, place 1 in column 'Alone'
test.loc[ test.FamilySize == 1, 'Alone' ] = 1

# plot both the variables
plt.figure(2, figsize=(20,15))
the_grid = plt.GridSpec(2, 2)

plt.subplot(the_grid[0, 0])
ax = sns.pointplot( x = "FamilySize", y = "Survived", data = train )
ax.set_xlabel('Family Size', fontsize=16, labelpad=10)
ax.set_title("Family size w.r.t survival rate", fontsize=18)
ax.set_ylabel("Survived", fontsize=16)
ax.tick_params(labelsize=14)

plt.subplot(the_grid[0, 1])
ax = sns.pointplot( x = "Alone", y = "Survived", data = train )
ax.set_xlabel('Alone', fontsize=16, labelpad=10)
ax.set_title("Alone w.r.t survival rate", fontsize=18)
ax.set_ylabel("Survived", fontsize=16)
ax.tick_params(labelsize=14)

plt.suptitle("Family Size and Alone", x=0.5, y=0.925, fontsize=20)
# For train data....
train['Has_Cabin'] = train.Cabin.apply( lambda x: 0 if type(x) == float else 1 )

# For test data....
test['Has_Cabin'] = test.Cabin.apply( lambda x: 0 if type(x) == float else 1 )
# For train data
train.loc[(train.Age.isnull())&(train.Titles=='Mr'),'Age']=33
train.loc[(train.Age.isnull())&(train.Titles=='Mrs'),'Age']=36
train.loc[(train.Age.isnull())&(train.Titles=='Master'),'Age']=5
train.loc[(train.Age.isnull())&(train.Titles=='Miss'),'Age']=22
train.loc[(train.Age.isnull())&(train.Titles=='Other'),'Age']=46

# For test data
test.loc[(test.Age.isnull())&(test.Titles=='Mr'),'Age']=33
test.loc[(test.Age.isnull())&(test.Titles=='Mrs'),'Age']=36
test.loc[(test.Age.isnull())&(test.Titles=='Master'),'Age']=5
test.loc[(test.Age.isnull())&(test.Titles=='Miss'),'Age']=22
test.loc[(test.Age.isnull())&(test.Titles=='Other'),'Age']=46
train.Embarked.fillna('S', inplace=True)
test.Embarked.fillna('S', inplace=True)
test.Fare.fillna(np.nanmean(test.Fare), inplace=True)
# For train data....
train.drop(['Name', 'PassengerId', 'Cabin', 'Ticket'], axis = 1, inplace = True)
# For test data....
test.drop(['Name', 'PassengerId', 'Cabin', 'Ticket'], axis = 1, inplace = True)
train.head()
# convert string values to numeric

train['Sex'].replace( ['male', 'female'], [0,1], inplace = True )
test.Sex.replace( ['male', 'female'], [0,1], inplace = True )

train['Titles'].replace( ['Mr', 'Mrs', 'Miss', 'Master', 'Other'], [0, 1, 2, 3, 4], inplace = True )
test.Titles.replace( ['Mr', 'Mrs', 'Miss', 'Master', 'Other'], [0, 1, 2, 3, 4], inplace = True )

train['Embarked'].replace( ['C', 'Q', 'S'], [0, 1, 2], inplace = True )
test.Embarked.replace( ['C', 'Q', 'S'], [0, 1, 2], inplace = True )
plt.figure(figsize = (15,12)) 
ax = sns.heatmap( train.corr(), annot=True, cmap = sns.diverging_palette(250, 10, as_cmap=True) )
ax.set_title("Heat map showing correlation between attributes")
train['Titles'].replace( [0, 1, 2, 3, 4], ['Mr', 'Mrs', 'Miss', 'Master', 'Other'],inplace = True )
test.Titles.replace( [0, 1, 2, 3, 4], ['Mr', 'Mrs', 'Miss', 'Master', 'Other'], inplace = True )

train['Embarked'].replace( [0, 1, 2], ['C', 'Q', 'S'], inplace = True )
test.Embarked.replace( [0, 1, 2], ['C', 'Q', 'S'], inplace = True )

dummies = pd.get_dummies(train[['Embarked', 'Titles']])
train = pd.concat([train, dummies], axis=1)
train.drop(['Embarked', 'Titles'], axis=1, inplace=True)

dummies = pd.get_dummies(test[['Embarked', 'Titles']])
test = pd.concat([test, dummies], axis=1)
test.drop(['Embarked', 'Titles'], axis=1, inplace=True)
train.head()
test.head()
required_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
       'FamilySize', 'Alone', 'Has_Cabin', 'Embarked_C', 'Embarked_Q',
       'Embarked_S', 'Titles_Master', 'Titles_Miss', 'Titles_Mr', 'Titles_Mrs',
       'Titles_Other']

class_label = 'Survived'

X_train = train[required_features].copy()
y_train = train[class_label].copy()
clf = DecisionTreeClassifier(random_state=0)
decision_tree_acc = cross_val_score( clf, X_train, y_train, cv=10, scoring = 'accuracy').mean()
print("Decision tree accuracy = ", decision_tree_acc)
clf.fit(X_train, y_train)
plt.figure(figsize = (14,7)) 
ax = sns.barplot(x=clf.feature_importances_, y=required_features)
max_leaf_nodes = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
cross_val_acc_max_leafs = []

for leaf in max_leaf_nodes:
    clf = DecisionTreeClassifier( max_leaf_nodes=leaf, random_state=0)
    cross_val_acc_max_leafs.append(cross_val_score(clf, X_train, y_train, cv=10, scoring = 'accuracy').mean())

depths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
cross_val_acc_depths = []

for depth in depths:
    clf = DecisionTreeClassifier( max_depth=depth, random_state=0)
    cross_val_acc_depths.append(cross_val_score(clf, X_train, y_train, cv=10, scoring = 'accuracy').mean())
    
min_sample_leafs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
cross_val_acc_min_samples = []

for leaf in min_sample_leafs:
    clf = DecisionTreeClassifier( min_samples_leaf=leaf, random_state=0)
    cross_val_acc_min_samples.append(cross_val_score(clf, X_train, y_train, cv=10, scoring = 'accuracy').mean())

min_sample_split = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
cross_val_acc_min_split = []

for leaf in min_sample_split:
    clf = DecisionTreeClassifier( min_samples_leaf=leaf, random_state=0)
    cross_val_acc_min_split.append(cross_val_score(clf, X_train, y_train, cv=10, scoring = 'accuracy').mean())

# plot both the variables
plt.figure(4, figsize=(20,15))
the_grid = plt.GridSpec(2, 2)

plt.subplot(the_grid[0, 0])
ax = sns.lineplot(max_leaf_nodes, cross_val_acc_max_leafs, color='b')
ax.set_xlabel('Max leaf nodes', fontsize=16, labelpad=10)
#ax.set_title("Family size w.r.t survival rate", fontsize=18)
ax.set_ylabel("Accuracy", fontsize=16)
ax.tick_params(labelsize=14)

plt.subplot(the_grid[0, 1])
ax = sns.lineplot(depths, cross_val_acc_depths, color='b')
ax.set_xlabel('Max depth', fontsize=16, labelpad=10)
#ax.set_title("Alone w.r.t survival rate", fontsize=18)
ax.set_ylabel("Accuracy", fontsize=16)
ax.tick_params(labelsize=14)

plt.subplot(the_grid[1, 0])
ax = sns.lineplot(min_sample_leafs, cross_val_acc_min_samples, color='b')
ax.set_xlabel('Min sample leafs', fontsize=16, labelpad=10)
#ax.set_title("Family size w.r.t survival rate", fontsize=18)
ax.set_ylabel("Accuracy", fontsize=16)
ax.tick_params(labelsize=14)

plt.subplot(the_grid[1, 1])
ax = sns.lineplot(min_sample_split, cross_val_acc_min_split, color='b')
ax.set_xlabel('Min sample split', fontsize=16, labelpad=10)
#ax.set_title("Alone w.r.t survival rate", fontsize=18)
ax.set_ylabel("Accuracy", fontsize=16)
ax.tick_params(labelsize=14)

plt.suptitle("Decision Tree Parameters", x=0.5, y=0.925, fontsize=20)
param_dist = {"criterion":['gini', 'entropy'],
              "splitter":['best', 'random'],
              "max_depth": sp_randint(1, 15),
              "min_samples_split": sp_randint(2, 20),
              "max_leaf_nodes": sp_randint(2, 20),
              "min_samples_leaf": sp_randint(2, 20),
             }

clf = DecisionTreeClassifier(random_state=0)
random_search = RandomizedSearchCV( estimator= clf, param_distributions=param_dist, n_iter=10, cv=10)
random_search.fit(X_train, y_train)
random_search.best_params_
clf = DecisionTreeClassifier(criterion= 'gini', max_depth= 9, max_leaf_nodes=4,
                             min_samples_leaf=17, min_samples_split=19, splitter='best')
tuned_decision_tree = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy').mean()
print("Accuracy after hyperparameter tuning : ", tuned_decision_tree)
clf = BaggingClassifier(random_state=0)
bagging_acc = cross_val_score( clf, X_train, y_train, cv=10, scoring = 'accuracy').mean()
print("Bagging classifier accuracy = ", bagging_acc)
clf = ExtraTreesClassifier(n_estimators=10, random_state=0)
extra_tree_acc = cross_val_score( clf, X_train, y_train, cv=10, scoring = 'accuracy').mean()
print("Extra Tree Classifier accuracy = ", extra_tree_acc)
clf = RandomForestClassifier(n_estimators=10, random_state=0)
random_forest_acc = cross_val_score( clf, X_train, y_train, cv=10, scoring = 'accuracy').mean()
print("Random forest accuracy = ", random_forest_acc)
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
ada_boost_acc = cross_val_score( clf, X_train, y_train, cv=10, scoring = 'accuracy').mean()
print("Ada Boost accuracy = ", ada_boost_acc)
clf = GradientBoostingClassifier(n_estimators=100, random_state=0)
gardient_boost_acc = cross_val_score( clf, X_train, y_train, cv=10, scoring = 'accuracy').mean()
print("Gradient Boost accuracy = ", gardient_boost_acc)
acc = [decision_tree_acc, bagging_acc, ada_boost_acc, gardient_boost_acc, random_forest_acc, extra_tree_acc]
classifiers = ['Decision Tree','Bagging Classifier','Ada Boost','Gradient Boost','Random Forest', 'Extra Tree Classifier']

sns.barplot(x=acc, y=classifiers)
