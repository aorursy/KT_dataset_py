import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 500)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

simplefilter(action='ignore', category=DeprecationWarning)

%matplotlib inline
# Load and check the datasets 

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

train_data.head()
test_data.head()
print('Shape of the training data: {}'.format(train_data.shape))

print('Shape of the testing data:  {}'.format(test_data.shape))
fig = plt.figure(figsize=(12,10))

sns.heatmap(train_data.iloc[:, 1:].corr(), annot=True)
submission = test_data['PassengerId']

train_data.drop(['PassengerId'], axis=1, inplace=True)

test_data.drop(['PassengerId'], axis=1, inplace=True)
def get_stats(df):

    lines = df.shape[0]

    d_types = df.dtypes

    count = df.count()

    unique = df.apply(lambda x: x.unique().shape[0])

    nulls = df.isnull().sum()

    null_ratio = (df.isnull().sum()/lines)*100

    skew = df.skew()

    names = ['data_types', 'count', 'unique', 'null', 'null_ratio']

    temp_df = pd.concat([d_types, count, unique, nulls, null_ratio], axis=1)

    temp_df.columns = names

    return temp_df

    

train_stats = get_stats(train_data)

train_stats
test_stats = get_stats(test_data)

test_stats
# since the information from the ticket number can probably be found in embarked and cabin we will get rid of that category as well



train_data.drop(['Ticket'], axis=1, inplace=True)

test_data.drop(['Ticket'], axis=1, inplace=True)
# there is one null in the Fare test data and 2 nulls in Embarked train data

test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
# now for cabin

train_data['Cabin'].fillna(0, inplace=True)

test_data['Cabin'].fillna(0, inplace=True)

train_data.loc[(train_data['Cabin'] != 0), 'Cabin'] = 1

test_data.loc[(test_data['Cabin'] != 0), 'Cabin'] = 1

print(train_data['Cabin'].value_counts())

print(test_data['Cabin'].value_counts())
# now for deling with name

train_data['Name'] = train_data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

test_data['Name'] = test_data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

    



train_title_mask = (train_data['Name'].value_counts() < 10)

test_title_mask = (test_data['Name'].value_counts() < 10)

train_data['Name'] = train_data['Name'].apply(lambda x: 'Misc' if train_title_mask.loc[x] == True else x)

test_data['Name'] = test_data['Name'].apply(lambda x: 'Misc' if test_title_mask.loc[x] == True else x)





print(train_data['Name'].value_counts())

print(test_data['Name'].value_counts())
# now we will create two variables family size and alone

train_data['F_size'] = train_data['Parch'] + train_data['SibSp'] + 1

test_data['F_size'] = test_data['Parch'] + test_data['SibSp'] + 1

train_data['Alone'] = 0

test_data['Alone'] = 0

train_data['Alone'].loc[(train_data['F_size'] == 1)] = 1

test_data['Alone'].loc[(test_data['F_size'] == 1)] = 1
# bar chart of cat cols

fig, axis = plt.subplots(2, 5, figsize=(20,10))



sns.countplot(x='Survived', data=train_data, ax=axis[0,0] )

axis[0,0].set_title('Survived')



sns.countplot(x='Pclass', hue='Survived', data=train_data, ax=axis[0,1])

axis[0,1].set_title('Pclass')



sns.countplot(x='Sex', hue='Survived', data=train_data, ax=axis[0,2])

axis[0,2].set_title('Sex')



sns.countplot(x='F_size', hue='Survived', data=train_data, ax=axis[0,3])

axis[0,3].set_title('F_size')



sns.countplot(x='Alone', hue='Survived', data=train_data, ax=axis[0,4])

axis[0,4].set_title('Alone')



sns.countplot(x='SibSp', hue='Survived', data=train_data, ax=axis[1,0])

axis[1,0].set_title('SibSp')



sns.countplot(x='Parch', hue='Survived', data=train_data, ax=axis[1,1])

axis[1,1].set_title('Parch')



sns.countplot(x='Embarked', hue='Survived', data=train_data, ax=axis[1,2])

axis[1,2].set_title('Embarked')



sns.countplot(x='Name', hue='Survived', data=train_data, ax=axis[1,3])

axis[1,3].set_title('Name')



sns.countplot(x='Cabin', hue='Survived', data=train_data, ax=axis[1,4])

axis[1,4].set_title('Cabin')
train_data.drop(['Parch', 'SibSp'], axis=1, inplace=True)

test_data.drop(['Parch', 'SibSp'], axis=1, inplace=True)
# since age is most correlated with Pclass we can set the age nan's with the median age grouped by Pclass

temp = train_data.groupby('Pclass')[['Age']].median()

temp





for i in range(3):

    train_data.loc[(train_data['Age'].isnull() & (train_data['Pclass'] == (i + 1))), 'Age'] = temp.loc[(i + 1), 'Age']

    test_data.loc[(test_data['Age'].isnull() & (test_data['Pclass'] == (i + 1))), 'Age'] = temp.loc[(i + 1), 'Age']

get_stats(train_data)
get_stats(test_data)
fig, ax = plt.subplots(2, 2, figsize=(16, 10))

sns.distplot(train_data['Age'], ax=ax[0, 0])

ax[0, 0].set_title('Train Age. skew = {}'.format(train_data['Age'].skew()))

sns.distplot(test_data['Age'], ax=ax[0, 1])

ax[0, 1].set_title('Test Age. skew = {}'.format(test_data['Age'].skew()))

sns.distplot(train_data['Fare'], ax=ax[1, 0])

ax[1, 0].set_title('Train Fare. skew = {}'.format(train_data['Fare'].skew()))

sns.distplot(test_data['Fare'], ax=ax[1, 1])

ax[1, 1].set_title('Test Fare. skew = {}'.format(test_data['Fare'].skew()))
# plot histogram for numeric columns

def plots(df, cols, coly='Survived', hist=True):

    for col in cols:

        fig, ax = plt.subplots(1, 2, figsize=(16, 5))

        sns.distplot(df[col], hist=hist, ax=ax[0])

        ax[0].set_title("density hist for " + col + '. skewdness =  %6.4f' %(df[col].skew()))

        ax[0].set_ylabel('frequency')

        sns.boxenplot(data=df, y=col, x=coly, ax=ax[1])

        plt.show()





numeric_cols = ['Fare', 'Age']

plots(train_data, numeric_cols)
Label = train_data['Survived'].values

train_data.drop(['Survived'], axis=1, inplace=True)

print(train_data.shape)

print(test_data.shape)
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

train_data['Sex'] = label.fit_transform(train_data['Sex'])

test_data['Sex'] = label.fit_transform(test_data['Sex'])

train_data.head()
train_data[['Name', 'Embarked', 'F_size']] = train_data[['Name', 'Embarked', 'F_size']].astype(str)

test_data[['Name', 'Embarked', 'F_size']] = test_data[['Name', 'Embarked', 'F_size']].astype(str)

print(train_data.shape)

train_data = pd.get_dummies(train_data)

test_data = pd.get_dummies(test_data)

print(train_data.shape)

print(test_data.shape)
from sklearn.preprocessing import PowerTransformer, StandardScaler

# now lets scale the data because Fare is skewed we will use the power transformer to minimize any effect from that

pow_scale = PowerTransformer(method='yeo-johnson').fit(train_data[['Fare']])

train_data[['Fare']] = pow_scale.transform(train_data[['Fare']])

test_data[['Fare']] = pow_scale.transform(test_data[['Fare']])



std_scale = StandardScaler().fit(train_data[['Age']])

train_data[['Age']] = std_scale.transform(train_data[['Age']])

test_data[['Age']] = std_scale.transform(test_data[['Age']])
train_data.head()
fig, ax = plt.subplots(2, 2, figsize=(16, 10))

sns.kdeplot(train_data['Fare'], shade=True, color='r', ax=ax[0,0])

ax[0,0].set_title('Training Data Fare')

sns.kdeplot(test_data['Fare'], shade=True, color='b', ax=ax[0,1])

ax[0,1].set_title('Testing Data Fare')

sns.kdeplot(train_data['Age'], shade=True, color='r', ax=ax[1,0])

ax[1,0].set_title('Training Data Age')

sns.kdeplot(test_data['Age'], shade=True, color='b', ax=ax[1,1])

ax[1,1].set_title('Testing Data Age')
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(train_data.values, Label)



features = train_data.columns

importance = rfc.feature_importances_

indicies = np.argsort(importance)



fig = plt.figure(figsize=(15, 12))

plt.barh(range(len(indicies)), importance[indicies], align='center')

plt.title('Feature importance')

plt.yticks(range(len(indicies)), [features[i] for i in indicies])

plt.xlabel('Importance')

plt.show()
# double check the shapes

train_data = train_data.values

test_data = test_data.values

print('Shape of the training data: {}'.format(train_data.shape))

print('Shape of the testing Data:  {}'.format(test_data.shape))

print('Shape of the Label:         {}'.format(Label.shape))
from sklearn.model_selection import KFold, cross_val_score

from sklearn.base import clone

# initial tests with models before tuning

def initial_score(mod, X, y, folds):

    cv = KFold(n_splits=folds, shuffle=True)

    cv_estimate = cross_val_score(mod, X, y, cv=cv, scoring='accuracy', n_jobs=4)

    mean = np.mean(cv_estimate)

    std = np.std(cv_estimate)

    return mean, std
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

svc_mod = SVC(probability=True)

ada_mod = AdaBoostClassifier()

rfc_mod = RandomForestClassifier()

gbc_mod = GradientBoostingClassifier()

mlp_mod = MLPClassifier()
names = ['SVC', 'AdaBoostClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'MLPClassifier']

mods = [svc_mod, ada_mod, rfc_mod, gbc_mod, mlp_mod]

results = pd.DataFrame(columns=['mean accuracy', 'std accuracy'], index=names)

for name, mod in zip(names, mods):

    mean, std = initial_score(mod, train_data, Label, 10)

    results.loc[name, 'mean accuracy'] = mean

    results.loc[name, 'std accuracy'] = std



results
from sklearn.model_selection import train_test_split, GridSearchCV

cv = KFold(n_splits=10, shuffle=True)

X_train, X_test, y_train, y_test = train_test_split(train_data, Label, test_size = 0.3)
svc_cf = SVC(probability=True)

svc_cf.get_params()
param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},

            {'C': [1, 10, 100, 1000], 'gamma': [1.0/50.0, 1.0/200.0, 1.0/500.0, 1.0/1000.0], 'kernel': ['rbf']}]



svc_class_grid = GridSearchCV(estimator=svc_cf,

                           param_grid=param_grid,

                           cv=cv,

                           scoring='accuracy',

                           return_train_score=True,

                           n_jobs=4,

                           verbose=1)



svc_class_grid.fit(train_data, Label)

svc_best_params = svc_class_grid.best_estimator_

print('GridsearchCV Best Score: {}'.format(svc_class_grid.best_score_))

print('\nTested HyperParameters           Values')

print('kernal:                             {}'.format(svc_class_grid.best_estimator_.kernel))

print('gamma:                              {}'.format(svc_class_grid.best_estimator_.gamma))

print('C:                                  {}'.format(svc_class_grid.best_estimator_.C))
from sklearn import metrics

svm_class = clone(svc_best_params)

svm_class.fit(X_train, y_train)

probabilities = svm_class.predict_proba(X_test)



def score_model(probs, threshold):

    return np.array([1 if x >= threshold else 0 for x in probs[:,1]])



def print_metrics(labels, probs, threshold):

    scores = score_model(probs, threshold)

    mets = metrics.precision_recall_fscore_support(labels, scores)

    conf = metrics.confusion_matrix(labels, scores)

    print('                 Confusion matrix')

    print('                 Score positive    Score negative')

    print('Actual positive    {:6d}             {:6d}'.format(conf[0,0], conf[0,1]))

    print('Actual negative    {:6d}             {:6d}'.format(conf[1,0], conf[1,1]))

    print('')

    print('Accuracy        {:.4f}'.format(metrics.accuracy_score(labels, scores)))

    print('AUC             {:.4f}'.format(metrics.roc_auc_score(labels, probs[:,1])))

    print('Macro precision {:.4f}'.format(float((float(mets[0][0]) + float(mets[0][1]))/2.0)))

    print('Macro recall    {:.4f}'.format(float((float(mets[1][0]) + float(mets[1][1]))/2.0)))

    print(' ')

    print('           Positive      Negative')

    print('Num case   {:6d}         {:6d}'.format(mets[3][0], mets[3][1]))

    print('Precision  {:.4f}         {:.4f}'.format(mets[0][0], mets[0][1]))

    print('Recall     {:.4f}         {:.4f}'.format(mets[1][0], mets[1][1]))

    print('F1         {:.4f}         {:.4f}'.format(mets[2][0], mets[2][1]))



def plot_auc(labels, probs):

    fpr, tpr, threshold = metrics.roc_curve(labels, probs[:,1])

    auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, color = 'blue', label = 'AUC = {:.4f}'.format(auc))

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()



print_metrics(y_test, probabilities, .5)      

plot_auc(y_test, probabilities)
ad_clf = AdaBoostClassifier(DecisionTreeClassifier())

ad_clf.get_params()
# Nested cross validation takes a while

param_grid = {'base_estimator__max_depth' :[1, 2, 3, 5],

              'base_estimator__min_samples_split':[2, 3, 5],

              'base_estimator__min_samples_leaf':[2, 3, 5, 10],

              'n_estimators' :[50, 100, 250, 500, 750],

              'learning_rate' :[0.0001, 0.001, 0.01, 0.1, 1, 5]}







ad_clf_grid = GridSearchCV(estimator=ad_clf,

                        param_grid=param_grid,

                        cv=cv,

                        scoring='accuracy',

                        return_train_score=True,

                        n_jobs=4,

                        verbose=1)



ad_clf_grid.fit(train_data, Label)

ada_best_params = ad_clf_grid.best_estimator_

print('GridsearchCV Best Score: ' + str(ad_clf_grid.best_score_))

print('\nTested HyperParameters           Values')

print('base_estimator__max_depth:           {}'.format(ad_clf_grid.best_estimator_.base_estimator.max_depth))

print('base_estimator__min_samples_split:   {}'.format(ad_clf_grid.best_estimator_.base_estimator.min_samples_split))

print('base_estimator__min_samples_leaf :   {}'.format(ad_clf_grid.best_estimator_.base_estimator.min_samples_leaf))

print('n_estimators:                        {}'.format(ad_clf_grid.best_estimator_.n_estimators))

print('learning_rate:                       {}'.format(ad_clf_grid.best_estimator_.learning_rate))
ad_clf_mod = clone(ada_best_params)

ad_clf_mod.fit(X_train, y_train)

probabilities = ad_clf_mod.predict_proba(X_test)



print_metrics(y_test, probabilities, .5)      

plot_auc(y_test, probabilities)
rf_clf = RandomForestClassifier()

rf_clf.get_params()
param_grid = {'n_estimators': [10, 50, 100, 250, 500, 750],

             'max_depth': [5, 8, 15, 25, 30],

             'min_samples_split': [2, 5, 10, 15],

             'min_samples_leaf':[2, 3, 5, 10, 15, 20]}



rf_clf_grid = GridSearchCV(estimator=rf_clf,

                        param_grid=param_grid,

                        cv=cv,

                        scoring='accuracy',

                        return_train_score=True,

                        n_jobs=4,

                        verbose=1)



rf_clf_grid.fit(train_data, Label)

rf_best_params = rf_clf_grid.best_estimator_

print('GridsearchCV Best Score: {}'.format(rf_clf_grid.best_score_))

print('\nTested HyperParameters   Values')

print('n_estimators:                {}'.format(rf_clf_grid.best_estimator_.n_estimators))

print('max_depth:                   {}'.format(rf_clf_grid.best_estimator_.max_depth))

print('min_samples_split:           {}'.format(rf_clf_grid.best_estimator_.min_samples_split))

print('min_samples_leaf:            {}'.format(rf_clf_grid.best_estimator_.min_samples_leaf))
rf_clf_mod = clone(rf_best_params)

rf_clf_mod.fit(X_train, y_train)

probabilities = rf_clf_mod.predict_proba(X_test)



print_metrics(y_test, probabilities, .5)      

plot_auc(y_test, probabilities)
gb_clf = GradientBoostingClassifier()

gb_clf.get_params()


param_grid = {'n_estimators': [50, 100, 250, 500, 750, 1000],

             'max_depth': [3, 5, 8, 15, 25],

             'min_samples_split': [2, 5, 10],

             'min_samples_leaf':[1, 2, 3, 5, 10],

             'learning_rate' :[0.0001, 0.001, 0.01, 0.1, 1]}



gb_clf_grid = GridSearchCV(estimator=gb_clf,

                        param_grid=param_grid,

                        cv=KFold(n_splits=5, shuffle=True),

                        scoring='accuracy',

                        return_train_score=True,

                        n_jobs=4,

                        verbose=1)



gb_clf_grid.fit(train_data, Label)

gb_best_params = gb_clf_grid.best_estimator_

print('GridsearchCV Best Score: {}'.format(gb_clf_grid.best_score_))

print('\nTested HyperParameters   Values')

print('n_estimators:                {}'.format(gb_clf_grid.best_estimator_.n_estimators))

print('max_depth:                   {}'.format(gb_clf_grid.best_estimator_.max_depth))

print('min_samples_split:           {}'.format(gb_clf_grid.best_estimator_.min_samples_split))

print('min_samples_leaf:            {}'.format(gb_clf_grid.best_estimator_.min_samples_leaf))

print('learning_rate:               {}'.format(gb_clf_grid.best_estimator_.learning_rate))
gb_clf_mod = clone(gb_best_params)

gb_clf_mod.fit(X_train, y_train)

probabilities = gb_clf_mod.predict_proba(X_test)



print_metrics(y_test, probabilities, .5)

plot_auc(y_test, probabilities)
mlp_clf = MLPClassifier()

mlp_clf.get_params()
param_grid = {'hidden_layer_sizes' :[(50, 50, 10), (50, 100, 10), (100,)],

             'solver' :['lbfgs', 'adam'],

             'alpha' :[.000001, .00001, .0001, .001],

             'early_stopping' :[True, False],

             'beta_1' :[.8, .9, .99],

             'beta_2' :[.8, .9, .99, .999],

             'max_iter' :[100, 200, 500, 1000]}



mlp_clf_grid = GridSearchCV(estimator=mlp_clf,

                           param_grid=param_grid,

                           cv=KFold(n_splits=5, shuffle=True),

                           scoring='accuracy',

                           return_train_score=True,

                           n_jobs=4,

                           verbose=1)



mlp_clf_grid.fit(train_data, Label)

mlp_best_params = mlp_clf_grid.best_estimator_

print('GridSearchCV Best Score:  {}'.format(mlp_clf_grid.best_score_))

print('\nTested HyperParameters   Values')

print('hidden_layer_sizes:          {}'.format(mlp_clf_grid.best_estimator_.hidden_layer_sizes))

print('solver:                      {}'.format(mlp_clf_grid.best_estimator_.solver))

print('alpha:                       {}'.format(mlp_clf_grid.best_estimator_.alpha))

print('early_stopping:              {}'.format(mlp_clf_grid.best_estimator_.early_stopping))

print('beta_1:                      {}'.format(mlp_clf_grid.best_estimator_.beta_1))

print('beta_2:                      {}'.format(mlp_clf_grid.best_estimator_.beta_2))

print('max_iter:                    {}'.format(mlp_clf_grid.best_estimator_.max_iter))
mlp_clf_mod = clone(mlp_best_params)

mlp_clf_mod.fit(X_train, y_train)

probabilities = mlp_clf_mod.predict_proba(X_test)



print_metrics(y_test, probabilities, .5)

plot_auc(y_test, probabilities)
from sklearn.ensemble import VotingClassifier

voting_cf = VotingClassifier(estimators=[('rfc', rf_best_params),

                                         ('svc', svc_best_params),

                                         ('ada', ada_best_params),

                                         ('gbc', gb_best_params), 

                                         ('mlp', mlp_best_params)], voting='soft', n_jobs=4)



voting_cf = voting_cf.fit(X_train, y_train)

voting = voting_cf.predict(X_test)
def print_metrics_score(labels, scores):

    mets = metrics.precision_recall_fscore_support(labels, scores)

    conf = metrics.confusion_matrix(labels, scores)

    print('                 Confusion matrix')

    print('                 Score positive    Score negative')

    print('Actual positive    {:6d}             {:6d}'.format(conf[0,0], conf[0,1]))

    print('Actual negative    {:6d}             {:6d}'.format(conf[1,0], conf[1,1]))

    print('')

    print('Accuracy        {:.4f}'.format(metrics.accuracy_score(labels, scores)))

    print('Macro precision {:.4f}'.format(float((float(mets[0][0]) + float(mets[0][1]))/2.0)))

    print('Macro recall    {:.4f}'.format(float((float(mets[1][0]) + float(mets[1][1]))/2.0)))

    print(' ')

    print('           Positive      Negative')

    print('Num case   {:6d}         {:6d}'.format(mets[3][0], mets[3][1]))

    print('Precision  {:.4f}         {:.4f}'.format(mets[0][0], mets[0][1]))

    print('Recall     {:.4f}         {:.4f}'.format(mets[1][0], mets[1][1]))

    print('F1         {:.4f}         {:.4f}'.format(mets[2][0], mets[2][1]))



    

print_metrics_score(y_test, voting)
from sklearn.model_selection import learning_curve

# this comes from sklearn plotting learning curves examples page

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html





def plot_learning_curve(estimator, title, X, y, axes, ylim=None, cv=None,

                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):



    axes.set_title(title)

    if ylim is not None:

        axes.set_ylim(*ylim)

    axes.set_xlabel("Training examples")

    axes.set_ylabel("Score")



    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)



    # Plot learning curve

    axes.grid(color='k')

    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,

                         train_scores_mean + train_scores_std, alpha=0.1,

                         color="r")

    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,

                         test_scores_mean + test_scores_std, alpha=0.1,

                         color="g")

    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",

                 label="Training score")

    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",

                 label="Cross-validation score")

    axes.legend(loc="best")

    return plt



cv=KFold(n_splits=10, shuffle=True)



fig, axes = plt.subplots(3, 2, figsize=(16, 30))





title = "Learning Curve (SVM Classifier)"

estimator = clone(svc_best_params)

plot_learning_curve(estimator, title, train_data, Label, axes=axes[0,0], ylim=(0.7, 1.01), cv=cv, n_jobs=4)



title = "Learning Curve (AdaBoostClassifier)"

estimator = clone(ada_best_params)

plot_learning_curve(estimator, title, train_data, Label, axes=axes[0,1], ylim=(0.7, 1.01), cv=cv, n_jobs=4)



title = "Learning Curve (RandomForestClassifier)"

estimator = clone(rf_best_params)

plot_learning_curve(estimator, title, train_data, Label, axes=axes[1,0], ylim=(0.7, 1.01), cv=cv, n_jobs=4)



title = "Learning Curve (GradientBoostingClassifier)"

estimator = clone(gb_best_params)

plot_learning_curve(estimator, title, train_data, Label, axes=axes[1,1], ylim=(0.7, 1.01), cv=cv, n_jobs=4)



title = "Learning Curve (MLPClassifier)"

estimator = clone(mlp_best_params)

plot_learning_curve(estimator, title, train_data, Label, axes=axes[2,0], ylim=(0.7, 1.01), cv=cv, n_jobs=4)



title = "Learning Curve (VotingClassifier)"

estimator = clone(voting_cf)

plot_learning_curve(estimator, title, train_data, Label, axes=axes[2,1], ylim=(0.7, 1.01), cv=cv, n_jobs=4)



plt.show()

svc_best = clone(svc_best_params)

svc_best.fit(train_data, Label)

scores = score_model(svc_best.predict_proba(test_data), .5)

sub = pd.DataFrame({'PassengerId':submission, 'Survived': scores})

sub.to_csv('svc_prediction.csv', index=False)
ada_best = clone(ada_best_params)

ada_best.fit(train_data, Label)

scores = score_model(ada_best.predict_proba(test_data), .5)

sub = pd.DataFrame({'PassengerId':submission, 'Survived': scores})

sub.to_csv('adaboost_prediction.csv', index=False)
rf_best = clone(rf_best_params)

rf_best.fit(train_data, Label)

scores = score_model(rf_best.predict_proba(test_data), .5)

sub = pd.DataFrame({'PassengerId':submission, 'Survived': scores})

sub.to_csv('random_forest_prediction.csv', index=False)
gb_best = clone(gb_best_params)

gb_best.fit(train_data, Label)

scores = score_model(gb_best.predict_proba(test_data), .5)

sub = pd.DataFrame({'PassengerId':submission, 'Survived': scores})

sub.to_csv('gradient_boost_prediction.csv', index=False)
mlp_best = clone(mlp_best_params)

mlp_best.fit(train_data, Label)

scores = score_model(mlp_best.predict_proba(test_data), .5)

sub = pd.DataFrame({'PassengerId':submission, 'Survived': scores})

sub.to_csv('MPL_nural_prediction.csv', index=False)
voting_final_cf = clone(voting_cf)

voting_final_cf.fit(train_data, Label)

sub = pd.DataFrame({'PassengerId':submission, 'Survived': voting_final_cf.predict(test_data)})

sub.to_csv('4_est_vote_prediction.csv', index=False)