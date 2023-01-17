# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style='white', color_codes=True)

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
# Importing the datasets

train = pd.read_csv('../input/titanic-datasets/train.csv')
test = pd.read_csv('../input/titanic-datasets/test.csv')
train.info()
train.isnull().sum()
test.info()
test.isnull().sum()
print(train.shape)
print(test.shape)
train.head()
sns.barplot('Pclass', 'Survived', data = train)
test.head()
train.describe()
# relation between features and survival
survived = train[train['Survived']==0]
not_survived = train[train['Survived']==1]

print("Survived: %i (%.1f%%)" % (len(survived), float(len(survived))/len(train)*100))
print("Not_Survived: %i (%.1f%%)" % (len(not_survived), float(len(not_survived))/len(train)*100))
print("Total: %i" % len(train))
# total passengers in different Pclass
train.Pclass.value_counts()
# survival in different Pclass
Pclass_survived = train.groupby('Pclass').Survived.value_counts()
Pclass_survived
Pclass_survived.unstack(level=0).plot(kind='bar',subplots=False)
Pclass_survived_avg = train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean()
Pclass_survived_avg
Pclass_survived_avg.plot(kind='bar', subplots=False)
# Number of males and females boarded
train.Sex.value_counts()
sns.barplot('Sex','Survived', data=train)
sex_survived = train.groupby('Sex').Survived.value_counts()
sex_survived
sex_survived.unstack(level=0).plot(kind='bar', subplots=False)
# seeing the survival based on Pclass
sns.factorplot('Sex', 'Survived', hue='Pclass', size=4, aspect=2, data=train)
sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=train)
train.Embarked.value_counts()
train.groupby('Embarked').Survived.value_counts()
sns.barplot('Embarked', 'Survived', data=train)
Embarked_survived_avg = train[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean()
Embarked_survived_avg
Embarked_survived_avg.plot(kind='bar', subplots=False)
# showing the survival rate based on Embarked, Pclass, Sex by violinplot

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

sns.violinplot('Embarked', 'Age', hue = 'Survived', data=train, split=True, ax=ax1)
sns.violinplot('Pclass', 'Age', hue = 'Survived', data=train, split=True, ax=ax2)
sns.violinplot('Sex', 'Age', hue = 'Survived', data=train, split=True, ax=ax3)
# Distribution plot based on survival rate

total_survived = train[train['Survived']==1]
total_not_survived = train[train['Survived']==0]

total_male_survived = train[(train['Survived']==1) & (train['Sex']=='male')]
total_female_survived = train[(train['Survived']==1) & (train['Sex']=='female')]

male_not_survived = train[(train['Survived']==0) & (train['Sex']=='male')]
female_not_survived = train[(train['Survived']==0) & (train['Sex']=='female')]

plt.figure(figsize=(15,5))
plt.subplot(111)

sns.distplot(total_survived['Age'].dropna().values, kde=True, bins=range(0,81,1), color='blue')
sns.distplot(total_not_survived['Age'].dropna().values, kde=True, bins=range(0,81,1), color='red', axlabel = 'Age')
plt.figure(figsize=(15,5))
plt.subplot(121)
sns.distplot(total_male_survived['Age'].dropna().values, kde=True, bins=range(0,81,1), color='blue')
sns.distplot(male_not_survived['Age'].dropna().values, kde=True, bins=range(0,81,1), color='red', axlabel= 'Male Age')

plt.subplot(122)
sns.distplot(total_female_survived['Age'].dropna().values, kde=True, bins=range(0,81,1), color='blue')
sns.distplot(female_not_survived['Age'].dropna().values, kde=True, bins=range(0,81,1), color='red', axlabel= 'Female Age')

# 1. Missing Values in train dataset
# 2. defining a function for missing values

def missing_value(train):
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total,percent], axis = 1, keys = ['Total', 'Percent'])
    return missing_data

missing_value(train)
def missing_value(test):
    total = test.isnull().sum().sort_values(ascending=False)
    percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total,percent], axis = 1, keys = ['Total', 'Percent'])
    return missing_data

missing_value(test)
train = train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
test = test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
# Average Age for different Pclass

train.groupby(['Pclass']).Age.mean()
test.groupby(['Pclass']).Age.mean()
# defining the function which will used later on for replacing missing values for Age.

def Age_approx(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 38
        elif Pclass==2:
            return 30
        else:
            return 25
    else:
        return Age
# defining the function which will used later on for replacing missing values for Age.

def age_approx(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 41
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(Age_approx, axis=1)
test['Age'] = test[['Age','Pclass']].apply(age_approx, axis=1)
train.isnull().sum()
# dropping the embarked missing value 
train.dropna(inplace=True)
test.isnull().sum()
test.dropna(inplace=True)
train.head()
test.head()
# Changing the categorical value for Sex and Embarked in both train and test dataset
# to avoid dummy trap, using drop_first=True

train_dummied = pd.get_dummies(train, columns=["Sex"], drop_first = True)
train_dummied = pd.get_dummies(train_dummied, columns=["Embarked"], drop_first = True)
test_dummied = pd.get_dummies(test, columns=["Sex"], drop_first=True)
test_dummied = pd.get_dummies(test_dummied, columns=["Embarked"], drop_first=True)
train_dummied.head()
test_dummied.head()
X = train_dummied.drop(['Survived'], axis=1)
y=train_dummied['Survived']
# Splitting the data

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler
independent_scalar = StandardScaler()
X_train = independent_scalar.fit_transform (X_train) #fit and transform
X_test = independent_scalar.transform (X_test) # only transform
test_dummied = independent_scalar.transform(test_dummied)

## Feature Scaling  and categorical encoding is not required for tree-based model model.
# Random Forest
# Decision Tree
# Adaboost
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
svc = SVC()
svc.fit(X_train,y_train)
y_pred_svc = svc.predict(X_test)
print(accuracy_score(y_test, y_pred_svc))
print(confusion_matrix(y_test, y_pred_svc))
linear_svc = LinearSVC()
linear_svc.fit(X_train,y_train)
y_pred_linear_svc = linear_svc.predict(X_test)

print(accuracy_score(y_test, y_pred_linear_svc))
print(confusion_matrix(y_test, y_pred_linear_svc))
clf = SGDClassifier(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_sgd = clf.predict(X_test)

print(accuracy_score(y_test, y_pred_sgd))
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print(accuracy_score(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred_dt = dt.predict(X_test)
confusion_matrix(y_test, y_pred_dt)
accuracy_score(y_test, y_pred_dt)
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)

print(confusion_matrix(y_test,y_pred_rf))
print(accuracy_score(y_test,y_pred_rf))
print(classification_report(y_test,y_pred_rf))
nb = GaussianNB()
nb.fit(X_train,y_train)
y_pred_nb = nb.predict(X_test)

print(accuracy_score(y_test,y_pred_nb))
print(classification_report(y_test,y_pred_nb))
abc = AdaBoostClassifier()
abc.fit(X_train,y_train)
y_pred_abc = abc.predict(X_test)
accuracy_score(y_test,y_pred_abc)
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
y_pred_xgb = xgb.predict(X_test)
accuracy_score(y_test, y_pred_xgb)
clf = Perceptron(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_perceptron = clf.predict(X_test)
accuracy_score(y_test,y_pred_perceptron)
# Importing the libraries

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
k_values = np.array([1,3,5,7,9,11,13,15])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsClassifier()
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 10, scoring = 'accuracy')
grid_result = grid.fit(X_train,y_train)

print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, sdv,param in zip(means, stds, params):
    print('%f (%f) with: %r' % (mean, sdv, param))

models = []
models.append(('LR', LogisticRegression()))
models.append(('SVC', SVC()))
models.append(('Linear_SVC', LinearSVC()))
models.append(('SGD', SGDClassifier(max_iter=5, tol=None)))
models.append(('NB', GaussianNB()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier(n_estimators = 100)))
models.append(('abc', AdaBoostClassifier()))
models.append(('Perceptron', Perceptron(max_iter=5, tol=None)))
models.append(('XGBoost', XGBClassifier()))
results =[]
names =[]
for name, model in models:
    cv_result = cross_val_score(model, X_train, y_train, cv = 10, scoring = 'accuracy')
    results.append(cv_result)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_result.mean(), cv_result.std()))
# defining Learning curve function

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
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

    plt.legend(loc="best")
    return plt
# defining validation curve function

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

# Plot learning curves LOGISTIC REGRESSION
title = "Learning Curves (Logistic Regression)"
plot_learning_curve(lr, title, X_train, y_train, ylim=(0.7, 1.01), cv=10, n_jobs=1);
# Plot validation curve lOGISTIC REGRESSION
title = 'Validation Curve (Logistic Regression)'
param_name = 'C'
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] 
cv = 10
plot_validation_curve(estimator=lr, title=title, X=X_train, y=y_train, param_name=param_name,
                      ylim=(0.5, 1.01), param_range=param_range);
# Plot learning curves with SVM

title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = 10
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator, title, X_train, y_train, ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)
# validation plot with SVM

title = 'Validation Curve (Naive Bayes)'
param_name = 'gamma'
param_range = np.logspace(-6,-1,5) 
cv = 10
plot_validation_curve(estimator=svc, title=title, X=X_train, y=y_train, param_name=param_name,
                      ylim=(0.5, 1.01), param_range=param_range);
# learning plot with NAIVE BAYES

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

title = "Learning Curves (Naive Bayes)"
plot_learning_curve(nb, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4);
fig = plt.figure(figsize = (15,8))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
y_valid_xgb = xgb.predict(test_dummied)
print(X.shape)
print(y.shape)
print(X_train.shape)
print(y_train.shape)
from sklearn.model_selection import RepeatedStratifiedKFold

model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# defining  the mode and parameters

model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
from sklearn.linear_model import RidgeClassifier

model = RidgeClassifier()
alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# define grid search
grid = dict(alpha=alpha)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

model = KNeighborsClassifier()
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']
# define grid search
grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

model = SVC()
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']
# define grid search
grid = dict(kernel=kernel,C=C,gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
from sklearn.ensemble import BaggingClassifier

# define models and parameters
model = BaggingClassifier()
n_estimators = [10, 100, 1000]
# define grid search
grid = dict(n_estimators=n_estimators)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# define models and parameters

model = RandomForestClassifier()
n_estimators = [10, 100, 1000]
max_features = ['sqrt', 'log2']
# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
from sklearn.ensemble import GradientBoostingClassifier

# define models and parameters
model = GradientBoostingClassifier()
n_estimators = [10, 100, 1000]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]
# define grid search
grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# defining model and parameters

model = XGBClassifier()
n_estimators = [1000]
learning_rate = [0.01, 0.1]
subsample = [0.8,1.0]
max_depth = [3]
colsample_bytree = [0.8, 0.9, 1.0]
gamma = [1]
# define grid search
grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth, 
            colsample_bytree = colsample_bytree, gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score='raise')
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

