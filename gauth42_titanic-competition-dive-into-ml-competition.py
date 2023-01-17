%matplotlib inline



# Data manipulation

import numpy as np

import pandas as pd



# Data visualization

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")



# Pre-processing

from sklearn import preprocessing



# Disable warnings

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
# Training set shape

print(train.shape)

train.head()
# Test set shape

print(test.shape)

test.head()
total = train.append(test)

total.info()
total.isnull().sum()
plt.figure(figsize = (10, 10))

sns.heatmap(train[["Survived", "Sex", "Age", "Pclass", "Fare", "SibSp", "Parch"]].corr(), annot=True, fmt = ".2f", cmap = "coolwarm")

plt.show()
plt.figure(figsize = (10, 5))

plt.title("Survival count")

sns.countplot(y='Survived' ,data=train, palette='coolwarm')

plt.show()
plt.figure(figsize = (10, 5))

plt.title("Survival and Sex")

sns.countplot(x='Survived', data=train, palette='coolwarm', hue='Sex')

plt.show()
plt.figure(figsize = (12, 8))

plt.title("Survival and Sex, Age")

sns.violinplot(x='Sex', y='Age', data=train, hue='Survived', split=True, palette='coolwarm')

plt.show()
f, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)





axes[0].set_title("Passenger class count ")

sns.countplot(y='Pclass', data=train, palette='coolwarm',  ax=axes[0])



axes[1].set_title("Passenger class count by Sex")

sns.countplot(y='Pclass', data=train, palette='coolwarm', hue='Sex',  ax=axes[1])



plt.show()
plt.figure(figsize = (10, 5))

plt.title("Survival and Passenger class")

sns.barplot(x='Pclass', y='Survived', data=train, palette='coolwarm', ci=None)

plt.show()
plt.figure(figsize = (10, 5))

plt.title("Survival and Passenger class, Sex")

sns.barplot(x='Pclass', y='Survived', data=train, palette='coolwarm', ci=None, hue='Sex')

plt.show()
plt.figure(figsize = (10, 7))

plt.title("Statistical distribution of Fare")

sns.kdeplot(train.Fare, shade=True)

plt.show()

train.Fare.describe()
plt.figure(figsize = (10, 7))

plt.title("Statistical distribution of Fare by Survival")



surv=train.Fare[train.Survived==1]

died=train.Fare[train.Survived==0]



sns.kdeplot(surv, label='Survived', shade=True)

sns.kdeplot(died, label='Did not survive', shade=True)

plt.show()
plt.figure(figsize = (10, 10))

plt.title("Survival and Passenger class, Fare")

sns.swarmplot(x='Pclass', y='Fare', data=train, hue='Sex')

plt.show()
f, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=False)



axes[0].set_title("Siblings/ spouses on the Titanic count")

sns.countplot(y='SibSp', data=train, palette='coolwarm', ax=axes[0])



axes[1].set_title("Parents/ children on the Titanic count")

sns.countplot(y='Parch', data=train, palette='coolwarm', ax=axes[1])



plt.show()
f, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)



axes[0].set_title("Siblings/ spouses on the Titanic and Survival")

sns.barplot(x='SibSp', y='Survived', data=train, palette='coolwarm', ci=None, ax=axes[0])



axes[1].set_title("Parents/ children on the Titanic count and Survival")

sns.barplot(x='Parch', y='Survived', data=train, palette='coolwarm', ci=None, ax=axes[1])



plt.show()
plt.figure(figsize = (10, 7))

plt.title("Embarkation port count")

sns.countplot(y='Embarked', data=train, palette='coolwarm')



plt.show()
train.Embarked.fillna('S', inplace=True)



plt.figure(figsize = (10, 7))

plt.title("Survival and Embarkation")

sns.barplot(x='Embarked', y='Survived', data=train, ci=None, palette='coolwarm')



plt.show()
train.Embarked.fillna('S', inplace=True)



plt.figure(figsize = (10, 7))

plt.title("Survival and Embarkation")

sns.countplot(x='Embarked', data=train, hue='Pclass', palette='coolwarm')



plt.show()
train.Name.to_frame().head()
split_fn = lambda x : x.split(',')[1].split('.')[0].strip()



# Extract titles

total['Title'] = total.Name.apply(split_fn)



# Plot titles

plt.figure(figsize = (20, 10))

plt.title("Titles count")

sns.countplot(y='Title', data=total, palette='coolwarm')



plt.show()
total.Title.unique()
total.replace(['Mlle', 'Ms'], 'Miss', inplace=True)

total.replace('Mme', 'Mrs', inplace=True)

total.replace(['Major', 'Col', 'Capt'], 'Military', inplace=True)

total.replace(['the Countess', 'Jonkheer', 'Dona', 'Don', 'Sir', 'Lady'], 'Fancy', inplace=True)
tot_mr = total[total.Title=='Mr']

tot_mrs = total[total.Title=='Mrs']

tot_miss = total[total.Title=='Miss']

tot_master = total[total.Title=='Master']

tot_fancy = total[total.Title=='Fancy']

tot_rev = total[total.Title=='Rev']

tot_dr = total[total.Title=='Dr']

tot_military = total[total.Title=='Military']
f, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=True)



axes[0].set_title("Title and Age")

sns.kdeplot(tot_mr.Age, label='Mr', shade=True, ax=axes[0])

sns.kdeplot(tot_mrs.Age, label='Mrs', shade=True, ax=axes[0])

sns.kdeplot(tot_miss.Age, label='Miss', shade=True, ax=axes[0])

sns.kdeplot(tot_master.Age, label='Master', shade=True, ax=axes[0])



axes[1].set_title("Title and Age")

sns.kdeplot(tot_fancy.Age, label='Fancy', shade=True, ax=axes[1])

sns.kdeplot(tot_rev.Age, label='Rev', shade=True, ax=axes[1])

sns.kdeplot(tot_dr.Age, label='Dr', shade=True, ax=axes[1])

sns.kdeplot(tot_military.Age, label='Military', shade=True, ax=axes[1])



plt.show()
total.loc[(total.Title=='Mr') & (np.isnan(total.Age)), 'Age'] = tot_mr.Age.median()

total.loc[(total.Title=='Mrs') & (np.isnan(total.Age)), 'Age'] = tot_mrs.Age.median()

total.loc[(total.Title=='Miss') & (np.isnan(total.Age)), 'Age'] = tot_miss.Age.median()

total.loc[(total.Title=='Master') & (np.isnan(total.Age)), 'Age'] = tot_master.Age.median()

total.loc[(total.Title=='Fancy') & (np.isnan(total.Age)), 'Age'] = tot_fancy.Age.median()

total.loc[(total.Title=='Rev') & (np.isnan(total.Age)), 'Age'] = tot_rev.Age.median()

total.loc[(total.Title=='Dr') & (np.isnan(total.Age)), 'Age'] = tot_dr.Age.median()

total.loc[(total.Title=='Military') & (np.isnan(total.Age)), 'Age'] = tot_military.Age.median()





total.Age.isnull().sum()
family_size = (total.SibSp + total.Parch).clip(0, 1)

total['is_alone'] = 1*(family_size==0) + 0
# Row with missing Fare value

total[total.Fare.isnull()]
total.loc[np.isnan(total['Fare']), 'Fare']  = total.loc[(total.Pclass==3) & (total.Embarked=='S')].Fare.median()
total['Fare'] = total['Fare'].apply(lambda x : np.log(x) if x>0 else 0)



plt.figure(figsize = (10, 7))

plt.title("Statistical distribution of Fare log")

sns.kdeplot(total.Fare, shade=True)

plt.show()

total.Fare.describe()
total.Cabin.describe()
total[total['Cabin'].notnull()].Cabin
total['Cabin'] = total['Cabin'].apply(lambda x: x[0] if not pd.isnull(x) else 'Z')
plt.figure(figsize = (10, 7))

plt.title("Survival and Embarkation")

sns.countplot(y='Cabin', data=total , palette='coolwarm')



plt.show()
# Create One-hot encoding df for each variables

dfDummies_embarked = pd.get_dummies(total['Embarked'], prefix='Embarked')

dfDummies_title = pd.get_dummies(total['Title'], prefix='Title')

dfDummies_sex = pd.get_dummies(total['Sex'], prefix='Sex')

dfDummies_cabin = pd.get_dummies(total['Cabin'], prefix='Cabin')



# Concate total wiht those df

total = pd.concat([total, dfDummies_embarked, dfDummies_title, dfDummies_sex, dfDummies_cabin],axis=1)



# Drop categorical features

total.drop(columns=['Embarked', 'Title', 'Sex', 'Cabin'], inplace=True, axis=1)



# Store PassengerId for the test set submission

passenger_id_test = total['PassengerId'][len(train):]



# Remove irrelevant columns for ML algorithms

total.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True, axis=1)
total.head()
#split total into train and test

train = total[:len(train)]

test = total[len(train):]



X = train.drop('Survived', axis = 1)

y = train['Survived'].astype(int)



X_test = test.drop('Survived', axis = 1)



#normalize X

X = preprocessing.StandardScaler().fit(X).transform(X)

X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)
import sklearn



from sklearn.model_selection import train_test_split

from sklearn import model_selection

from sklearn import metrics



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC



from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve





import tensorflow as tf
kfold = StratifiedKFold(n_splits=10)
random_state = 0



# Instantiate a list of classifiers

classifiers = []



# Decision Trees

classifiers.append(DecisionTreeClassifier(criterion='gini', random_state=random_state))

classifiers.append(DecisionTreeClassifier(criterion='entropy', random_state=random_state))

# Extra Trees

classifiers.append(ExtraTreesClassifier(criterion='gini', random_state=random_state))

classifiers.append(ExtraTreesClassifier(criterion='entropy', random_state=random_state))

# Random Forests

classifiers.append(RandomForestClassifier(criterion='gini', random_state=random_state))

classifiers.append(RandomForestClassifier(criterion='entropy', random_state=random_state))

# AdaBoost

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state, algorithm='SAMME.R', learning_rate=0.1))

# Logistic Regression

classifiers.append(LogisticRegression(random_state=random_state))

# K Nearest Neighbors

classifiers.append(KNeighborsClassifier())

# SVC

classifiers.append(SVC(kernel='linear'))

classifiers.append(SVC(kernel='rbf'))



algorithms = ['DecisionTree_gini', 'DecisionTree_entropy', 'ExtraTrees_gini', 'ExtraTrees_entropy', 'RandomForestClassifier_gini', 'RandomForestClassifier_entropy',

             'AdaBoostClassifier', 'LogisticRegression', 'KNeighborsClassifier', 'SVC_linear', 'SVC_rbf']
cv_results = []

cv_mean = []

cv_std = []



for classifier in classifiers:

    cv_results.append(cross_val_score(classifier, X, y=y, scoring='accuracy', cv=kfold, n_jobs=-1))



for cv_res in cv_results:

    cv_mean.append(cv_res.mean())

    cv_std.append(cv_res.std())

    

res_df = pd.DataFrame({"Cv_means": cv_mean, "Cv_errors": cv_std, "ML Algorithm": algorithms})





plt.figure(figsize = (10, 7))

plt.title("Cross Validation scores")

sns.barplot(x='Cv_means', y="ML Algorithm", data=res_df , palette='coolwarm', **{'xerr':cv_std})

plt.show()
dtree = DecisionTreeClassifier()

ada_dtree = AdaBoostClassifier(dtree)



ada_dtree_param_grid = {'base_estimator__criterion': ["gini", "entropy"],

                    "base_estimator__splitter" : ["best", "random"],

                    "learning_rate" : [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],

                    "algorithm" : ["SAMME","SAMME.R"],

                    "n_estimators" : [1,2],

                    }



gs_ada_dtree = GridSearchCV(ada_dtree, param_grid=ada_dtree_param_grid, cv=kfold, scoring='accuracy', n_jobs=-1, verbose=1)



gs_ada_dtree.fit(X, y)



ada_dtree_best = gs_ada_dtree.best_estimator_
gs_ada_dtree.best_score_
svm = SVC(probability=True)



svm_param_grid = {'kernel': ["rbf"],

                  'C' : [1, 10, 30, 100, 300, 500, 1000],

                  'gamma' : [0.001, 0.01, 0.1, 1.0]

                    }



gs_svm = GridSearchCV(svm, param_grid=svm_param_grid, cv=kfold, scoring='accuracy', n_jobs=-1, verbose=1)



gs_svm.fit(X, y)



svm_best = gs_svm.best_estimator_
gs_svm.best_score_
rf = RandomForestClassifier()



rf_param_grid = {'criterion' : ["gini"],

                 'max_features' : [1, 3, 5, 10],

                 'min_samples_split' : [2, 3, 10],

                 'min_samples_leaf': [1, 3, 10],

                 'bootstrap': [False],

                 'n_estimators' : [100, 300]

                }



gs_rf = GridSearchCV(rf, param_grid=rf_param_grid, cv=kfold, scoring='accuracy', n_jobs=-1, verbose=1)



gs_rf.fit(X, y)



rf_best = gs_rf.best_estimator_
gs_rf.best_score_
ex_tree = ExtraTreesClassifier()



ex_tree_param_grid = {'criterion' : ["gini"],

                 'max_features' : [1, 3, 5, 10],

                 'min_samples_split' : [2, 3, 10],

                 'min_samples_leaf': [1, 3, 10],

                 'bootstrap': [False],

                 'n_estimators' : [100, 300]

                }



gs_ex_tree = GridSearchCV(ex_tree, param_grid=ex_tree_param_grid, cv=kfold, scoring='accuracy', n_jobs=-1, verbose=1)



gs_ex_tree.fit(X, y)



ex_tree_best = gs_ex_tree.best_estimator_
gs_ex_tree.best_score_
def learning_curves(estimator, title, X, y, cv=None, n_jobs=-1):

    plt.figure()

    

    plt.title(title)

    plt.xlabel("Number of training examples")

    plt.ylabel("Accuracy")

    

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs)

    

    train_mean = np.mean(train_scores, axis=1)

    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)

    test_std = np.std(test_scores, axis=1)

    

    plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.1, color='b')

    plt.fill_between(train_sizes, test_mean-test_std, test_mean+test_std, alpha=0.1, color='r')

    

    plt.plot(train_sizes, train_mean, 'o-', color="b",label="Training accuracy")

    plt.plot(train_sizes, test_mean, 'o-', color="r",label="Testing accuracy")

    

    plt.legend(loc="best")

    return plt



classifiers = [(ada_dtree_best, "AdaBoost with Decision Trees"), (svm_best, "SVM"), (rf_best, "Random Forests"),

              (ex_tree_best, "Extra Trees")]



for classifier, name in classifiers:

    learning_curves(classifier, name, X, y, cv=kfold)
fig, axes = plt.subplots(1, 3, figsize=(25, 15), sharex="all")



num_class = 0



classifiers = [(ada_dtree_best, "AdaBoost with Decision Trees"),(rf_best, "Random Forests"), (ex_tree_best, "Extra Trees")]

for i in range(3):

    name = classifiers[num_class][1]

    classifier = classifiers[num_class][0]



    indices = np.argsort(classifier.feature_importances_)[::-1][:20]

    graph = sns.barplot(y=train.columns[indices][:20], 

                        x=classifier.feature_importances_[indices][:20] , orient='h',ax=axes[i])



    graph.set_title(name)

    graph.set_xlabel("Importance")

    graph.set_ylabel("Features")

    num_class += 1
test_rf = pd.Series(rf_best.predict(X_test), name="RFC")

test_ex_tree = pd.Series(ex_tree_best.predict(X_test), name="EXTC")

test_svm = pd.Series(svm_best.predict(X_test), name="SVM")

test_ada_dtree = pd.Series(ada_dtree_best.predict(X_test), name="Ada")



results = pd.concat([test_rf, test_ex_tree, test_svm, test_ada_dtree],axis=1)



plt.figure(figsize = (10, 10))

sns.heatmap(results.corr(), annot=True, cmap = "coolwarm")

plt.show()
voting = VotingClassifier(estimators=[("AdaBoost with Decision Trees", ada_dtree_best,), ("SVM", svm_best), 

                                      ("Random Forests", rf_best), ("Extra Trees", ex_tree_best)], voting='soft', n_jobs=-1)



voting = voting.fit(X, y)
train_acc = metrics.accuracy_score(voting.predict(X), y)



print("Accuracy on the training set : {:.3f}".format(train_acc))
test_pred = pd.Series(voting.predict(X_test), name="Survived")

results = pd.concat([passenger_id_test ,test_pred],axis=1)



results.head()
results.to_csv("results_titanic.csv",index=False)