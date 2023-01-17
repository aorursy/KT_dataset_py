# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# sklearn
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import RFECV

# to remove future warnings in pandas
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print(train_df.columns.values)
# preview the data
train_df.head()
print("TRAIN INFO:")
train_df.info()
print('#' * 40)
print("TEST INFO:")
test_df.info()
train_df.describe()
train_df.describe(include=['O'])
colormap = plt.cm.coolwarm
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_df.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
# class
print(train_df.pivot_table(index='Pclass', columns='Sex', values='Survived'))
# siblings & spouses
print(train_df.pivot_table(index='SibSp', columns='Sex', values='Survived'))
# parents & children
print(train_df.pivot_table(index='Parch', columns='Sex', values='Survived'))
# age
plt.figure(figsize=(14,12))
g = sns.FacetGrid(train_df, row='Survived', col='Sex', size=4)
g.map(sns.distplot, "Age", color="b")
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Age distribution with distinction to two sexes and survivability.".upper())

plt.show()
# fare 
plt.figure(figsize=(14,12))
g = sns.FacetGrid(train_df, row='Survived', col='Sex', size=6, palette="Set1")
g.map(sns.distplot, "Fare")
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Fare distribution with distinction to two sexes and survivability.".upper())
plt.show()
# looking at Sex
plt.figure(figsize=(14,12))
g = sns.FacetGrid(train_df, palette='coolwarm')
g.map(sns.barplot , 'Sex' , 'Survived', color="r")
# g.add_legend()
plt.show()
# Looking at Embarked
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train_df);
plt.show()
# Concatenate both data frames for calculations
both_df = pd.concat([train_df.drop('Survived', axis=1), test_df]) # we drop Survived from train
print("Shapes: ",train_df.drop('Survived', axis=1).shape, test_df.shape, both_df.shape)
print('_'*40)

# Fill missing values of Age with mean
print("Age mean = {}".format(both_df.Age.mean()))
both_df['Age'] = both_df.Age.fillna(both_df.Age.mean())
#Fill missing values of Fare with mean
print("Fare mean = {}".format(both_df.Fare.mean()))
both_df['Fare'] = both_df.Fare.fillna(both_df.Fare.mean())
# check what value of Embarked is the most frequent
(both_df.groupby('Embarked')).Embarked.count()
#Fill missing values of Embarked with 'S' for unknown
both_df['Embarked'] = both_df.Embarked.fillna('S')
#Fill missing values of Cabin with 'U0' for unknown
both_df['Cabin'] = both_df.Cabin.fillna('U0')
both_df.info()
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html
both_df['Title'] = both_df['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
pd.crosstab(both_df['Title'], both_df['Sex'])
pd.crosstab(both_df['Title'], train_df['Survived']).head(1)
both_df['Title'] = both_df['Title'].replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
both_df['Title'] = both_df['Title'].replace('Mlle', 'Miss')
both_df['Title'] = both_df['Title'].replace('Ms', 'Miss')
both_df['Title'] = both_df['Title'].replace('Mme', 'Mrs')
pd.crosstab(both_df['Title'], both_df['Sex'])
train_df["Title"] = both_df.loc[both_df.PassengerId <= train_df.shape[0], 'Title']
test_df["Title"] = both_df.loc[both_df.PassengerId > train_df.shape[0], 'Title']

sns.barplot(x="Title", y="Survived", data=train_df);
plt.title("Percent of survived in groups of different titles.")
# we create a ne column for the size of family + 1 (each one is member of its own family)
both_df['Family'] = both_df['Parch'] + both_df['SibSp'] + 1
# checking for correlation with survival rate
tmp_df = both_df.loc[both_df.PassengerId < len(train_df)]
tmp_df['Survived'] = train_df['Survived']
sns.barplot(x="Family", y="Survived", data=tmp_df);
plt.title("Percent of survived in groups of different family size.")
# we create three classes of families
family_df = pd.DataFrame()
family_df['FamilySize'] = both_df['Family']
family_df['PassengerId'] = both_df['PassengerId']
# tmp_df.loc[both_df['Family'] == 1, 'FamilySize'] = 'Alone'
# tmp_df.loc[both_df['Family'] > 4, 'FamilySize'] = 'Big'
# tmp_df.loc[tmp_df.FamilySize!='Alone' and tmp_df.FamilySize!='Big'] = 'Small'

family_df['FamilySize']  = both_df['Family'].map(lambda x :
                                              'Alone' if x == 1 
                                              else 'Big' if x>4
                                              else 'Small')
# visualizing
tmp_df = family_df.loc[both_df.PassengerId < len(train_df)]
tmp_df['Survived'] = train_df['Survived']
sns.barplot(x="FamilySize", y="Survived", data=tmp_df);
plt.title("Percent of survived in groups of different family size.")
plt.show()
# cAdding 'FamilySize' to data
both_df['FamilySize'] = family_df['FamilySize']
both_df.head(3)
cabin_df = pd.DataFrame()
cabin_df['CabinClass'] = both_df['Cabin'].map(lambda cabin: cabin[0:1])
cabin_df.groupby('CabinClass').size()
# checking for correlations:
tmp_df = cabin_df
tmp_df['PassengerId'] = both_df['PassengerId']
tmp_df = tmp_df.loc[tmp_df.PassengerId < len(train_df)]
tmp_df['Survived'] = train_df['Survived']
# visualization
sns.barplot(x="CabinClass", y="Survived", data=tmp_df, color='r');
plt.title("Percent of survived in groups of different cabin class.")
plt.show()
# we will keep that info
both_df['CabinClass']=cabin_df['CabinClass']
both_df.head(3)
# let's see Age distribution again
plt.figure(figsize=(10,4))
sns.distplot(both_df.Age, hist=True, bins=20)
ax = plt.gca()
ax.set_xticks(range(0,81,5))
plt.title('Age distribution')
plt.show()
# using the histogram we define bins to create a discrete variable
bins = [0, 12, 17, 24, 40, 60, np.inf]
labels = ['child', 'teen', 'young_adult', 'adult', 'late_adult','elder']
cp = both_df.copy()
age_types = pd.cut(cp.Age, bins, labels=labels)
cp['AgeGroup'] = age_types
cp.head(3)
# looking at the survival rate in train_df
age_types = pd.cut(train_df.Age, bins, labels=labels)
train_df[['Survived']].groupby(age_types).aggregate(np.mean)
# moving from copy to data
both_df['AgeGroup'] = cp['AgeGroup']
# adult males were most likely to die
def is_adult_male(row):
    if (row.AgeGroup == 'adult' or row.AgeGroup == 'young_adult') and row.Sex == 'male':
        return 1
    else:
        return 0
both_df['AdultMale'] = both_df.apply(is_adult_male, axis=1)
both_df.head(3)
# let's see Fare distribution again
plt.figure(figsize=(10,4))
sns.distplot(both_df.Fare, hist=True, bins=40)
ax = plt.gca()
plt.title('Fare distribution')
plt.show()
# using the histogram we define bins to create a discrete variable
bins = [0, 13, 20, 40, 100, np.inf]
labels = ['very_low', 'low', 'average', 'large', 'very large']
fare_types = pd.cut(cp.Fare, bins, labels=labels)
cp['FareGroup'] = fare_types
# looking at the survival rate in train_df
fare_types2 = pd.cut(train_df.Fare, bins, labels=labels)
train_df[['Survived']].groupby(fare_types2).aggregate(np.mean)
# moving from copy to data
both_df['FareGroup'] = cp['FareGroup']
# we will use pd.get_dummies function for interesting categorical features
both_df['Sex'] = pd.get_dummies(both_df.Sex)
both_df['Embarked'] = pd.get_dummies(both_df.Embarked)
both_df['Title'] = pd.get_dummies(both_df.Title)
both_df['FamilySize'] = pd.get_dummies(both_df.FamilySize)
both_df['CabinClass'] = pd.get_dummies(both_df.CabinClass)
both_df['AgeGroup'] = pd.get_dummies(both_df.AgeGroup)
both_df['FareGroup'] = pd.get_dummies(both_df.FareGroup)
# lets look what features we have
both_df.columns.values
# we don't need Name, Age, SibSp, Parch, Fare, Cabin, Family anymore
both_df = both_df.drop(['Name', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Family'],  axis=1)
# Ticket information is unnecessary
both_df = both_df.drop('Ticket', axis=1)
both_df.head(3)
# let's make new train and test data
new_train_df = both_df.loc[both_df.PassengerId <= len(train_df)]
new_train_df['Survived'] = train_df['Survived']
new_train_df = new_train_df.drop('PassengerId', axis=1)
new_test_df = both_df.loc[both_df.PassengerId > len(train_df)]
new_test_df = new_test_df.drop('PassengerId', axis=1)
print(train_df.shape, new_train_df.shape, test_df.shape, new_test_df.shape)
new_train_df.head(3)
test_X = new_test_df
train_X = new_train_df.drop('Survived', axis=1)
train_y = new_train_df.Survived
# first let's create lists of results for different models
metric_res = []
# list of classifiers
classifiers = []
# define scoring methods
scoring = ['accuracy', 'f1']
# logistic regression
log_reg = LogisticRegression()
classifiers.append(('log_reg', log_reg))
scores = cross_validate(log_reg, train_X, train_y, scoring=scoring,
                         cv=10, return_train_score=False)

acc = round(np.mean(scores['test_accuracy']),3)
acc_u = round(2*np.std(scores['test_accuracy']),3)
f1 = round(np.mean(scores['test_f1']),3)
f1_u = round(2*np.std(scores['test_f1']),3)
print("Accuracy: {} +/- {}, F1 score: {} +/- {}".format(acc,acc_u, f1, f1_u))
metric_res.append(("Logistic regression", acc, acc_u, f1, f1_u))
# support vector machines
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kk in kernels:
    svc = svm.SVC(kernel=kk, probability=True)
    classifiers.append(("svc_"+kk, svc))
    scores = cross_validate(svc, train_X, train_y, scoring=scoring,
                             cv=10, return_train_score=False)
    acc = round(np.mean(scores['test_accuracy']),3)
    acc_u = round(2*np.std(scores['test_accuracy']),3)
    f1 = round(np.mean(scores['test_f1']),3)
    f1_u = round(2*np.std(scores['test_f1']),3)
    print("Accuracy: {} +/- {}, F1 score: {} +/- {}".format(acc,acc_u, f1, f1_u))
    metric_res.append(("SVC "+kk, acc, acc_u, f1, f1_u))
# decision tree
tree_m = tree.DecisionTreeClassifier(max_depth=None)
classifiers.append(('tree', tree_m))
scores = cross_validate(tree_m, train_X, train_y, scoring=scoring,
                         cv=10, return_train_score=False)
acc = round(np.mean(scores['test_accuracy']),3)
acc_u = round(2*np.std(scores['test_accuracy']),3)
f1 = round(np.mean(scores['test_f1']),3)
f1_u = round(2*np.std(scores['test_f1']),3)
print("Accuracy: {} +/- {}, F1 score: {} +/- {}".format(acc,acc_u, f1, f1_u))
metric_res.append(("Decision tree", acc, acc_u, f1, f1_u))
# Gaussian Naive Bayes
gaussian = GaussianNB()
classifiers.append(('gnb', gaussian))
scores = cross_validate(gaussian, train_X, train_y, scoring=scoring,
                         cv=10, return_train_score=False)
acc = round(np.mean(scores['test_accuracy']),3)
acc_u = round(2*np.std(scores['test_accuracy']),3)
f1 = round(np.mean(scores['test_f1']),3)
f1_u = round(2*np.std(scores['test_f1']),3)
print("Accuracy: {} +/- {}, F1 score: {} +/- {}".format(acc,acc_u, f1, f1_u))
metric_res.append(("Gaussian Naive Bayes", acc, acc_u, f1, f1_u))
# Perceptron
perceptron = Perceptron()
classifiers.append(('perceptron', perceptron))
scores = cross_validate(perceptron, train_X, train_y, scoring=scoring,
                         cv=10, return_train_score=False)
acc = round(np.mean(scores['test_accuracy']),3)
acc_u = round(2*np.std(scores['test_accuracy']),3)
f1 = round(np.mean(scores['test_f1']),3)
f1_u = round(2*np.std(scores['test_f1']),3)
print("Accuracy: {} +/- {}, F1 score: {} +/- {}".format(acc,acc_u, f1, f1_u))
metric_res.append(("Perceptron", acc, acc_u, f1, f1_u))
# Stochastic Gradient Descent
sgd = SGDClassifier()
classifiers.append(('gradient',sgd))
scores = cross_validate(sgd, train_X, train_y, scoring=scoring,
                         cv=10, return_train_score=False)
acc = round(np.mean(scores['test_accuracy']),3)
acc_u = round(2*np.std(scores['test_accuracy']),3)
f1 = round(np.mean(scores['test_f1']),3)
f1_u = round(2*np.std(scores['test_f1']),3)
print("Accuracy: {} +/- {}, F1 score: {} +/- {}".format(acc,acc_u, f1, f1_u))
metric_res.append(("Gradient", acc, acc_u, f1, f1_u))
# RandomForest
# http://scikit-learn.org/stable/modules/ensemble.html#random-forests
rf = RandomForestClassifier(n_estimators=10,max_depth=None)
classifiers.append(('rf',rf))
scores = cross_validate(rf, train_X, train_y, scoring=scoring,
                         cv=10, return_train_score=False)
acc = round(np.mean(scores['test_accuracy']),3)
acc_u = round(2*np.std(scores['test_accuracy']),3)
f1 = round(np.mean(scores['test_f1']),3)
f1_u = round(2*np.std(scores['test_f1']),3)
print("Accuracy: {} +/- {}, F1 score: {} +/- {}".format(acc,acc_u, f1, f1_u))
metric_res.append(("Random forest", acc, acc_u, f1, f1_u))
# Extremely Randomized Trees
# http://scikit-learn.org/stable/modules/ensemble.html#extremely-randomized-trees
xtree = ExtraTreesClassifier(n_estimators=10, max_depth=None)
classifiers.append(('xtree',xtree))
scores = cross_validate(xtree, train_X, train_y, scoring=scoring,
                         cv=10, return_train_score=False)
acc = round(np.mean(scores['test_accuracy']),3)
acc_u = round(2*np.std(scores['test_accuracy']),3)
f1 = round(np.mean(scores['test_f1']),3)
f1_u = round(2*np.std(scores['test_f1']),3)
print("Accuracy: {} +/- {}, F1 score: {} +/- {}".format(acc,acc_u, f1, f1_u))
metric_res.append(("Extremely Randomized Trees", acc, acc_u, f1, f1_u))
models = pd.DataFrame({'Model': [x[0] for x in metric_res],
                       'Acc': [round(x[1],3) for x in metric_res],
                       'Acc_u': [round(x[2],3) for x in metric_res],
                       'F1': [round(x[3],3) for x in metric_res],
                       'F1_u': [round(x[3],3) for x in metric_res]})
models.head
models.sort_values(by='Acc', ascending=False)

# Recursive feature elimination and cross-validated selection of the best number of features
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
t_X, v_X, t_y, v_y = train_test_split(train_X, train_y, train_size = .7)

rfecv = RFECV(estimator = xtree, step = 1, cv = StratifiedKFold(t_y , 10), scoring = 'accuracy' )
rfecv.fit(t_X, t_y)

print (rfecv.score(t_X , t_y) , rfecv.score(v_X , v_y))
print( "Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected" )
plt.ylabel("Cross validation score" )
plt.plot(range(1, len( rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
# see what features were selected by the algorithm
sel_feat = train_X.columns[rfecv.support_]
print(sel_feat)
# take subset of selected data
reduced_train_X = train_X[sel_feat]
reduced_train_X.head(1)
# do the same for test data
reduced_test_X = test_X[sel_feat]
sel_xtree = ExtraTreesClassifier(max_depth=None, n_estimators=10, criterion='entropy')
scores = cross_validate(sel_xtree, reduced_train_X, train_y, scoring=scoring,
                         cv=10, return_train_score=False)
acc = round(np.mean(scores['test_accuracy']),3)
acc_u = round(2*np.std(scores['test_accuracy']),3)
f1 = round(np.mean(scores['test_f1']),3)
f1_u = round(2*np.std(scores['test_f1']),3)
print("Accuracy: {} +/- {}, F1 score: {} +/- {}".format(acc,acc_u, f1, f1_u))
# perform search for the best params
from sklearn.model_selection import GridSearchCV
parameters = {'criterion':('entropy', 'gini'), 'n_estimators':[10, 30]}
xt = ExtraTreesClassifier(max_depth=None)
grid = GridSearchCV(xt, parameters)
grid.fit(reduced_train_X, train_y)
print(grid.best_params_)
# check metrics for the improved model
scores = cross_validate(grid, reduced_train_X, train_y, scoring=scoring,
                         cv=10, return_train_score=False)
acc = round(np.mean(scores['test_accuracy']),3)
acc_u = round(2*np.std(scores['test_accuracy']),3)
f1 = round(np.mean(scores['test_f1']),3)
f1_u = round(2*np.std(scores['test_f1']),3)
print("Accuracy: {} +/- {}, F1 score: {} +/- {}".format(acc,acc_u, f1, f1_u))
# final predictions
sel_xtree = ExtraTreesClassifier(max_depth=None,
                                 n_estimators=grid.best_params_['n_estimators'],
                                 criterion=grid.best_params_['criterion'])
sel_xtree.fit(reduced_train_X, train_y)
final_pred = sel_xtree.predict(reduced_test_X)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": final_pred
    })
submission.to_csv('submission.csv', index=False)
