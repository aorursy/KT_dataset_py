# Standard librarie

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

sb.set(context = 'notebook', style = 'white', palette = 'deep')



# Ignore warning

import warnings

warnings.filterwarnings("ignore")



from collections import Counter



# Models libraries

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression, Perceptron

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split, learning_curve, GridSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score
# Load data

train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")

# Get number of raws in train and test sets

train_n = train_data.shape[0]

test_n = test_data.shape[0]

# ID of the passengers in the test set. We will use it at the end for submission of the results.

PassengerId_test = test_data["PassengerId"]

print('Number of training data: {}'.format(train_n))

print('Number of testing data: {}'.format(test_n))





# Combine train and test data into a dataset. In this way we can plot the behaviour of the whole dataset

# First check that test_data and test_data have the same fields, apart from the "Survived" field

print('Train fields : {}'.format(set(train_data.columns)))

print('Train fields : {}'.format(set(test_data.columns)))

dataset = pd.concat(objs = [train_data,test_data], axis = 0, ignore_index = True , sort = False)

print('Dataset fields : {}'.format(set(dataset.columns)))
# Fill empty and NaNs values with NaN

dataset = dataset.fillna(np.nan)



# Check for Null values

print('Number of null data in the dataset: \n{}'.format(dataset.isnull().sum()))

print(' ')

print(dataset.dtypes)
print(dataset.head(5))
plt.figure(figsize=[4,4])

plt.plot(dataset['PassengerId'])
# Drop the PassengerId feature from the dataset

dataset.drop(labels = ['PassengerId'], axis = 1, inplace = True)
survived_n = len(train_data[train_data['Survived'] == 1])



print('Percentage of survived passengers: {:.2f}.\nNot survived {:.2f}'.format(survived_n/train_n, 1-survived_n/train_n))
fig = plt.figure(figsize = [8,6])

sb.heatmap(train_data[['Survived','Age','Pclass','SibSp','Parch','Fare']].corr(), annot = True, fmt = '.2f', cmap = 'coolwarm')
fig = plt.figure(figsize = [12,10])

plt.subplot(331)

sb.kdeplot(train_data['Age'][(train_data['Survived']==0) & (train_data['Age'].notnull())], color = 'r', shade = True)

sb.kdeplot(train_data['Age'][(train_data['Survived']==1) & (train_data['Age'].notnull())], color = 'b', shade = True)

plt.xlabel('Age'); plt.ylabel('Survival probability')

plt.legend(['Not Survived','Survived'])



plt.subplot(332)

sb.barplot('Sex', 'Survived', data=train_data)



plt.subplot(333)

sb.barplot('Pclass', 'Survived', data=train_data)



plt.subplot(334)

sb.barplot('Embarked', 'Survived', data=train_data)



plt.subplot(335)

sb.barplot('SibSp', 'Survived', data=train_data)



plt.subplot(336)

sb.barplot('Parch', 'Survived', data=train_data)



plt.subplot(313)

sb.kdeplot(train_data['Fare'][(train_data['Survived']==0) & (train_data['Fare'].notnull())], color = 'r', shade = True)

sb.kdeplot(train_data['Fare'][(train_data['Survived']==1) & (train_data['Fare'].notnull())], color = 'b', shade = True)

plt.xlabel('Fare'); plt.ylabel('Survival probability')

plt.legend(['Not Survived','Survived'])



plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
# Fare distribution skewness

print('Skewness dataset: {}'.format(dataset['Fare'].skew()))

# Reduce Fare skewness by log function

dataset['Fare'] = dataset['Fare'].map(lambda x: np.log(x) if x>0 else 0)

# Observe new Fare distribution skewness

print('Skewness dataset after log transformation: {}'.format(dataset['Fare'].skew()))
fig = plt.figure(figsize = [8,6])

plt.subplot(311)

sb.kdeplot(dataset['Fare'][(dataset['Survived']==0) & (dataset['Fare'].notnull())], color = 'r', shade = True)

sb.kdeplot(dataset['Fare'][(dataset['Survived']==1) & (dataset['Fare'].notnull())], color = 'b', shade = True)

plt.xlabel('Fare')

plt.legend(['Not Survived','Survived'])



plt.subplot(323)

sb.barplot(x = 'SibSp', y = 'Survived', hue = 'Pclass', data = train_data)

plt.subplot(324)

sb.countplot(x = 'SibSp', hue = 'Pclass', data = train_data)



plt.subplot(325)

sb.countplot(x = 'Embarked', data = train_data)



plt.subplot(326)

sb.countplot(x = 'Embarked', hue = 'Pclass', data = train_data)



plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.55, wspace=0.35)
# Fill in the missing values in the embarked feature

print('Number of null data in the train_data: \n{}'.format(train_data.isnull().sum()))
train_data['Embarked'][(train_data['Embarked'].notnull())].value_counts()
# The 2 missing values we observed at the beginning are in the train_data.

# Therefore, we fill them with 'S', which is the most recurrent value.

dataset["Embarked"] = dataset['Embarked'].fillna('S')
test_data['Fare'] = test_data['Fare'].map(lambda x: np.log(x) if x>0 else 0)

dataset['Fare'] = dataset['Fare'].fillna(test_data['Fare'].median())
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})
# Method for replacing missing values in a DataFrame

def replace_missing_age(df):

    index_NaN_age = list(df['Age'][df['Age'].isnull()].index)

    for i in index_NaN_age :

        age_med = df['Age'].median()

        age_pred = df['Age'][((df['SibSp'] == df.iloc[i]['SibSp']) & (df['Parch'] == df.iloc[i]['Parch']) & (df['Pclass'] == df.iloc[i]['Pclass']))].median()

        if not np.isnan(age_pred) :

            df['Age'].iloc[i] = age_pred

        else :

            df['Age'].iloc[i] = age_med



# Replace missing values in the train_data

replace_missing_age(train_data)

# Replace missing values in the test_data

replace_missing_age(test_data)



# Replace train_data['Age'] and test_data['Age'] in the dataset

dataset['Age'][0:train_n] = train_data['Age']

dataset['Age'][train_n:] = test_data['Age']



# Plot new Age distribution

fig = plt.figure(figsize = [8,6])

sb.kdeplot(train_data['Age'][(train_data['Survived']==0) & (train_data['Age'].notnull())], color = 'r', shade = True)

sb.kdeplot(train_data['Age'][(train_data['Survived']==1) & (train_data['Age'].notnull())], color = 'b', shade = True)

plt.xlabel('Age'); plt.ylabel('Survival probability')

plt.legend(['Not Survived','Survived'])
unknown_cabin_data = dataset['Cabin'][dataset['Cabin'].isnull()].index

print('Number of missing Cabin data: {}/{} ({:.2f}%)'.format(len(unknown_cabin_data),len(dataset),100*len(unknown_cabin_data)/len(dataset)))



known_cabin_data = dataset['Cabin'][dataset['Cabin'].notnull()].index

print(dataset['Cabin'][known_cabin_data[:6]])
dataset['Cabin'][known_cabin_data] = dataset['Cabin'][known_cabin_data].str[0]

dataset['Cabin'][unknown_cabin_data] = 'U'



print(dataset['Cabin'][:7])
tab = pd.crosstab(dataset['Cabin'], dataset['Survived'])

tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.xlabel('Cabin')

plt.ylabel('Percentage')



fig = plt.figure(figsize = [12,6])

plt.subplot(121)

sb.countplot(x = 'Cabin', hue = 'Pclass', data = dataset)

plt.subplot(122)

sb.barplot(x = 'Cabin', y = 'Survived', hue = 'Pclass', data = dataset)
print('Number of null data: \n{}'.format(dataset.isnull().sum()))
titles = [name.split(',')[1].split('.')[0].strip() for name in dataset['Name']]

dataset['Title'] = pd.Series(titles)



cross = pd.crosstab(dataset['Title'], dataset['Survived'])

title_count = Counter(dataset['Title'])

print(cross)

print(' ')

print(title_count)
dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Dr', 'Jonkheer', 'Major', 'Sir', 'the Countess'], 'Wealth')

dataset['Title'] = dataset['Title'].replace(['Don', 'Rev', 'Dona'], 'Religious')

dataset['Title'] = dataset['Title'].replace(['Lady', 'Miss', 'Mlle', 'Mme', 'Mrs', 'Ms'], 'Women')



dataset['Title'] = dataset['Title'].map({'Master':0, 'Women': 1, 'Mr':2, 'Religious':3, 'Wealth':4})

dataset['Title'] = dataset['Title'].astype('int64')

# Plot the new Title distribution

cross = pd.crosstab(dataset['Title'], dataset['Survived'])

print(cross)
# Remove the name feature

dataset.drop(labels = 'Name', axis = 1, inplace = True)
dataset['Fsize'] = dataset['SibSp'] + dataset['Parch'] + 1



for idx,val in enumerate(dataset['Fsize']):

    if val == 1:

        dataset['Fsize'][idx] = 0

    elif val == 2:

        dataset['Fsize'][idx] = 1

    elif val > 2 and val <5:

        dataset['Fsize'][idx] = 2

    elif val >= 5:

        dataset['Fsize'][idx] = 3
print('Unique ticket: {} out of {}.'.format(dataset['Ticket'].nunique(),dataset['Ticket'].count()))

print('Number of alone passengers: {}'.format(dataset['Fsize'][dataset['Fsize']==0].count()))



print('Tickets counter {}'.format(Counter(dataset['Ticket'])))
group_ticket = train_data.groupby('Ticket')

k = 0

for name, group in group_ticket:

    if (len(group_ticket.get_group(name)) > 1):

        print(group.loc[:,['Survived','Name', 'Fare']])

        k += 1

    if (k>6):

        break
group_ticket_name = dataset.groupby('Ticket')['Fare'].transform('count')

dataset['Ticket_shared'] = np.where(dataset.groupby('Ticket')['Fare'].transform('count') > 1, 1, 0)
dataset['Ticket'] = dataset['Ticket'].str[0]



sb.catplot(x = 'Ticket', y = 'Survived', data = dataset, kind = 'bar')
dataset['Ticket_surv_rate'] = dataset['Ticket'].isin(['A','3','7','W','4','L','6','5','8']).astype('int64')
dataset.dtypes
# Fsize

dataset = pd.get_dummies(dataset, columns = ['Fsize'], prefix='Fam')

# Ticket

dataset = pd.get_dummies(dataset, columns = ['Ticket'], prefix='Tic')

# Cabin

dataset = pd.get_dummies(dataset, columns = ['Cabin'], prefix='Cab')

# Embarked

dataset = pd.get_dummies(dataset, columns = ['Embarked'], prefix='Emb')

# Title

dataset = pd.get_dummies(dataset, columns = ['Title'], prefix='Tit')
# Get training and testing dataset

train_data = dataset[:train_n]

test_data = dataset[train_n:]

test_data.drop(labels = ['Survived'], axis = 1, inplace = True)



# Separate the target from the train_data

train_data['Survived'] = train_data['Survived'].astype('int64')

y_train = train_data['Survived']

X_train = train_data.drop(labels = 'Survived', axis = 1)
random_state = 333

classifiers = []

classifiers.append(LogisticRegression(random_state = random_state))

classifiers.append(LinearDiscriminantAnalysis())

classifiers.append(Perceptron(random_state=random_state))

classifiers.append(MLPClassifier(random_state=random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(SVC(random_state=random_state))

classifiers.append(GaussianNB())

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(ExtraTreesClassifier(random_state=random_state))

classifiers.append(GradientBoostingClassifier(random_state=random_state))

classifiers.append(xgb.XGBClassifier())
# Kfold stratified for Cross validation

kfold = StratifiedKFold(n_splits = 10)



# Collect accuracy for all the classifiers

cv_results = []

for clf in classifiers:

    cv_results.append(cross_val_score(clf, X_train, y_train, cv = kfold, scoring = 'accuracy'))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



# Group results in a variable

cv_res = pd.DataFrame({'CrossValMeans':cv_means,'CrossValStd': cv_std,

                       'Algorithm':['LogisticRegression','LinearDiscriminantAnalysis','Perceptron',

                                    'MLPClassifier','KNeighborsClassifier','SVC','GaussianNB',

                                    'DecisionTreeClassifier','AdaBoostClassifier','RandomForestClassifier',

                                    'ExtraTreesClassifier','GradientBoostingClassifier','XGBClassifier']})
fig = plt.figure()

sb.barplot(x = 'CrossValMeans', y = 'Algorithm', data = cv_res, orient = 'h', **{'xerr':cv_std} )

plt.xlabel('Mean Accuracy')

plt.title('Cross validation scores')
# Random forest

RFC = RandomForestClassifier()

# Search grid

rf_param_grid = {'max_depth': [None],

              'max_features': [1, 3, 10],

              'min_samples_split': [2, 3, 10],

              'min_samples_leaf': [1, 3, 10],

              'bootstrap': [False],

              'n_estimators' :[100,300],

              'criterion': ['gini','entropy']}



gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring='accuracy', n_jobs= 4)

gsRFC.fit(X_train,y_train)

# Get the best model

RFC_best = gsRFC.best_estimator_



# ----------------------------------------------

# ----------------------------------------------



# ExtraTreesClassifier

ExtC = ExtraTreesClassifier()

# Search grid

ex_param_grid = {'max_depth': [None],

              'max_features': [1, 3, 10],

              'min_samples_split': [2, 3, 10],

              'min_samples_leaf': [1, 3, 10],

              'bootstrap': [False],

              'n_estimators' :[100,300],

              'criterion': ['gini','entropy']}



gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring='accuracy', n_jobs= 4)

gsExtC.fit(X_train,y_train)

# Get the best model

ExtC_best = gsExtC.best_estimator_



# ----------------------------------------------

# ----------------------------------------------



# Ada Boost Classifier

AdaDTC = AdaBoostClassifier(DecisionTreeClassifier(), random_state=7)

# Search grid

AdaDTC_param_grid = {'base_estimator__criterion' : ['gini','entropy'],

              'base_estimator__splitter' : ['best', 'random'],

              'algorithm' : ['SAMME'],

              'n_estimators' : [1,2],

              'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}



gsAdaDTC = GridSearchCV(AdaDTC,AdaDTC_param_grid,scoring = 'accuracy',n_jobs=4, cv = kfold)

gsAdaDTC.fit(X_train,y_train)

# Get the best model

Ada_best = gsAdaDTC.best_estimator_



# ----------------------------------------------

# ----------------------------------------------



# Gradient boosting tunning

GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ['deviance'],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring='accuracy', n_jobs= 4)

gsGBC.fit(X_train,y_train)

# Get the best model

GBC_best = gsGBC.best_estimator_
# Yassine's code

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

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

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



plot_learning_curve(RFC_best,'RF learning curves',X_train,y_train,cv=kfold)

plot_learning_curve(ExtC_best,'ExtraTrees learning curves',X_train,y_train,cv=kfold)

plot_learning_curve(Ada_best,'AdaBoost learning curves',X_train,y_train,cv=kfold)

plot_learning_curve(GBC_best,'GradientBoosting learning curves',X_train,y_train,cv=kfold)
import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')

    print(cm)



    fig = plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
# Check best RF Classifier parameters

RFC_best.get_params
# Split the data in training and validation sets

X_train_cm, X_valid_cm, y_train_cm, y_valid_cm = train_test_split(X_train, y_train, test_size=0.33, random_state=random_state)

# Random forest

RFC_cm = RandomForestClassifier(max_depth=None, max_features=10, max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=3, min_samples_split=10,

                       min_weight_fraction_leaf=0.0, n_estimators=300,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False)

RFC_cm.fit(X_train_cm,y_train_cm)



# Compute confusion matrix

cnf_matrix = confusion_matrix(RFC_cm.predict(X_valid_cm),y_valid_cm)



# Plot non-normalized and normalized confusion matrixes

titles_options = [('Confusion matrix, without normalization', None),

                  ('Normalized confusion matrix', 'true')]

for title, normalize in titles_options:

    disp = plot_confusion_matrix(cnf_matrix,['Not Survived', 'Survived'],

                                 normalize=normalize,

                                 title=title)
# Let's collect the scores from the best classifiers we optimised

RFC_score = gsRFC.best_score_

ExtC_score = gsExtC.best_score_

Ada_score = gsAdaDTC.best_score_

GBC_score = gsGBC.best_score_



scores = pd.DataFrame({'Model': ['RandomForest', 'ExtraTree', 'AdaBoost', 'GradientBoosting'],

                       'Score': [RFC_score, ExtC_score, Ada_score, GBC_score]})



chart = sb.barplot(x = 'Model', y = 'Score', data = scores)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

for clf,scr in zip(scores['Model'],scores['Score']):

    print('Score for {} : {:.4f}'.format(clf,scr))
fig = plt.figure(figsize=[12,12])



classifiers_info = [('RandomForest', RFC_best),

                    ('ExtraTree', ExtC_best),

                    ('AdaBoost', Ada_best),

                    ('GradientBoosting', GBC_best)]



for fig_idx, clf_info in enumerate(classifiers_info):

        plt.subplot(2,2,fig_idx+1)

        clf_name = clf_info[0]

        clf = clf_info[1]

        indices = np.argsort(clf.feature_importances_)[::-1][:20]

        sb.barplot(y = X_train.columns[indices][:40],x = clf.feature_importances_[indices][:40] , orient='h')

        plt.xlabel('Relative importance',fontsize=12)

        plt.ylabel('Features',fontsize=12)

        plt.title(clf_info[0] + ' feature importance')

        

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.45)
# Set up the voting classifier

votingClf = VotingClassifier(estimators=[('RandomForest', RFC_best),

                                       ('ExtraTree', ExtC_best),

                                       ('AdaBoost', Ada_best),

                                       ('GradientBoosting', GBC_best)], voting='soft', n_jobs=4)



votingClf_score = cross_val_score(votingClf, X_train, y_train, cv = kfold, scoring = 'accuracy')

print('Voting classifier score: {:.4f} (+/- {:.2f})'.format(votingClf_score.mean(), votingClf_score.std()))

# Class to extend the Sklearn classifier; this basically unifies the way we call each classifier 

class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

# Function for out-of-fold prediction

def get_oof(clf, x_train, y_train, x_test, kf, NFOLDS):

    ntrain = x_train.shape[0]

    ntest = x_test.shape[0]

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))

    

    # split data in NFOLDS training vs testing samples

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):

        # select train and test sample

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]

        

        # train classifier on training sample

        clf.train(x_tr, y_tr)

        

        # predict classifier for testing sample

        oof_train[test_index] = clf.predict(x_te)

        # predict classifier for original test sample

        oof_test_skf[i, :] = clf.predict(x_test)

    

    # take the median of all NFOLD test sample predictions

    # (changed from mean to preserve binary classification)

    oof_test[:] = np.median(oof_test_skf,axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
# Put in our parameters for selected classifiers

# Random Forest parameters

rf_params = {

    'n_estimators': 500,

     'warm_start': True, 

     #'max_features': 0.2,

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

}



# Extra Trees Parameters

et_params = {

    'n_estimators':500,

    #'max_features': 0.5,

    'max_depth': 8,

    'min_samples_leaf': 2,

}



# AdaBoost parameters

ada_params = {

    'n_estimators': 500,

    'learning_rate' : 0.75

}



# Gradient Boosting parameters

gb_params = {

    'n_estimators': 500,

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 2,

}



# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'linear',

    'C' : 0.025

    }
# Create objects for each classifier

rf = SklearnHelper(clf=RandomForestClassifier, seed=random_state, params=rf_params)

et = SklearnHelper(clf=ExtraTreesClassifier, seed=random_state, params=et_params)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=random_state, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=random_state, params=gb_params)

svc = SklearnHelper(clf=SVC, seed=random_state, params=svc_params)
# Create our OOF train and test predictions. These base results will be used as new features

X_test_st = test_data.values

X_train_st = X_train.values

y_train_st = y_train.values



NFOLDS = 5

kf = KFold(n_splits=NFOLDS, random_state=random_state)

et_oof_train, et_oof_test = get_oof(et, X_train_st, y_train_st, X_test_st, kf, NFOLDS)

rf_oof_train, rf_oof_test = get_oof(rf, X_train_st, y_train_st, X_test_st, kf, NFOLDS)

ada_oof_train, ada_oof_test = get_oof(ada, X_train_st, y_train_st, X_test_st, kf, NFOLDS)

gb_oof_train, gb_oof_test = get_oof(gb, X_train_st, y_train_st, X_test_st, kf, NFOLDS)

svc_oof_train, svc_oof_test = get_oof(svc, X_train_st, y_train_st, X_test_st, kf, NFOLDS)



print("Training is complete")
base_predictions_train = pd.DataFrame({'RandomForest': rf_oof_train.ravel(),

                                       'ExtraTrees': et_oof_train.ravel(),

                                       'AdaBoost': ada_oof_train.ravel(),

                                       'SVM' : svc_oof_train.ravel(),

                                       'GradientBoost': gb_oof_train.ravel()})

base_predictions_train.head()
plt.figure(figsize=(12,10))

foo = sb.heatmap(base_predictions_train.corr(), vmax=1.0, square=True, annot=True)
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)

x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
clf_stack = xgb.XGBClassifier(

    n_estimators= 2000,

    max_depth= 4,

    min_child_weight= 2,

    gamma=0.9,

    subsample=0.8,

    colsample_bytree=0.8,

    objective= 'binary:logistic',

    scale_pos_weight=1)
scores = cross_val_score(clf_stack, x_train, y_train, cv=5)

print('Stacking classifier score: {:.4f} (+/- {:.2f})'.format(np.mean(scores), np.std(scores)))
clf_voting = votingClf.fit(X_train,y_train)

voting_pred = clf.predict(test_data)



clf_stack = clf_stack.fit(x_train, y_train)

stack_pred = clf_stack.predict(x_test)



results = pd.DataFrame({'PassengerId': PassengerId_test,

#                         'Survived': voting_pred.T})

                        'Survived': stack_pred.T})



# Let's give a rapid look to the submit format

results.head(10)
# Submit

results.to_csv("../working/TitanicChallenge.csv", index=False)