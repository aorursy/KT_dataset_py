# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
%pprint

# tweaks for above libraries
# Numpy/Pandas & Matplotlib tweaks
# tweaks for Numpy & Pandas (see respective documentation)
pd.set_option('display.notebook_repr_html',True)
pd.set_option('display.max_rows',15)
pd.set_option('display.max_columns',25)
pd.set_option('display.width',1024)
# force all Numpy & Pandas floating point output to 3 decimal places
float_formatter = lambda x: '%.4f' % x
np.set_printoptions(formatter={'float_kind':float_formatter})
pd.set_option('display.float_format', float_formatter)
# force Numpy to display very small floats using floating point notation
np.set_printoptions(threshold=np.inf, suppress=True, precision=4, linewidth=2048)
# force GLOBAL floating point output to 4 decimal places
%precision 4

# tweaks for Seaborn plotting library
sns.set_style('darkgrid')
plt.style.use('seaborn-muted')
sns.set_style({'font.sans-serif':['Verdana','Arial','Calibri','DejaVu Sans']})
sns.set_context('talk')

seed = 42
np.random.seed(seed)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data_file, test_data_file = '../input/train.csv', '../input//test.csv'
train_df = pd.read_csv(train_data_file, index_col=0)
test_df = pd.read_csv(test_data_file, index_col=0)
len(train_df), len(test_df)
# view some loaded records
train_df.head()
# test_df.head()
train_df.info()
# describe numeric columns
train_df.describe()
#test_df.describe()
# and the categorical columns
train_df.describe(include=['O'])
#test_df.describe(include=['O'])
# do we have NULLs?
train_df.isnull().sum()
#test_df.isnull().sum()
# viewing survivor counts
_ = sns.countplot(x='Survived',data=train_df)
plt.show()
plt.close()  # to release plotting resources and reduce memory consumption!
# chance of survival by gender
survived_by_gender = train_df[['Sex','Survived']].groupby(['Sex']).mean().sort_values(by='Survived',ascending=False)
survived_by_gender
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,3))

_ = sns.countplot(x='Survived', data=train_df, ax=ax[0])
ax[0].set_title('Count of Survivors')
_ = sns.barplot(x='Sex', y='Survived', data=train_df, ax=ax[1])
ax[1].set_title('Chances of Survival by Gender')

plt.show()
plt.close()
# chance of survival by Pclass
train_df[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False)
_ = sns.barplot(x='Pclass',y='Survived',data=train_df)
_ = sns.barplot(x='Pclass',y='Survived', hue='Sex', data=train_df)
# chance of survival by age group
# lets bin the data into age-groups of 10 - we will add an "AgeBins" column
train_df['AgeBins'] = pd.cut(train_df.Age,bins=10,include_lowest=True)
survived_by_agebins = train_df[['AgeBins', 'Survived']].groupby(['AgeBins']).mean().sort_values(by='Survived', ascending=False)
survived_by_agebins
survived_by_agebins.sort_index(ascending=True).plot(kind='bar',figsize=(10,5))
survived_by_agebins_sex = \
    train_df[['AgeBins', 'Sex', 'Survived']].groupby(['AgeBins','Sex']).mean().sort_values(by='Survived', ascending=False)
survived_by_agebins_sex
f, ax = plt.subplots(figsize=(20,8))
_ = sns.barplot(x='AgeBins',y='Survived', hue='Sex', data=train_df,ax=ax)
plt.legend(loc='best')
plt.show()
plt.close()
f, ax = plt.subplots(figsize=(20,8))
_ = sns.barplot(x='Sex',y='Survived', hue='AgeBins', data=train_df,ax=ax)
plt.legend(loc='best')
plt.show()
plt.close()
# lets drop additional column we added just for analysis (keeping memory consumption down)
train_df.drop(['AgeBins'],axis=1,inplace=True) # drop col we had introduced for viewing only
del survived_by_agebins
del survived_by_agebins_sex
# chance of survival by SibSp
train_df[['SibSp', 'Survived']].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False)
_ = sns.barplot(x='SibSp',y='Survived',data=train_df)
# chance of survival by Parch
train_df[['Parch', 'Survived']].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False)
_ = sns.barplot(x='Parch',y='Survived',data=train_df)
familyOnBoard = lambda row : 1 if ((row.SibSp > 0) or (row.Parch > 0)) else 0 

train_df['Fob'] = train_df.apply(familyOnBoard, axis=1)
test_df['Fob'] = test_df.apply(familyOnBoard, axis=1)

train_df[['Survived','SibSp','Parch','Fob']].head()
#train_df[['Survived','SibSp','Parch','Fob']].head()
# chances of survival by Family on board
train_df[['Fob', 'Survived']].groupby(['Fob']).mean().sort_values(by='Survived', ascending=False)
_ = sns.barplot(x='Fob',y='Survived',data=train_df)
# chance of survival by port of embarkation
train_df[['Embarked', 'Survived']].groupby(['Embarked']).mean().sort_values(by='Survived', ascending=False)
# plot of chance of survival by port of embarkation & PClass
plt.figure(figsize=(8,6))
_ = sns.barplot(x='Embarked', y='Survived', hue='Pclass', data=train_df)
plt.show()
plt.close()
# let us extract the title (Mr. Miss. Mrs. etc.) from the name & save it in a new title column
train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_df[['Name','Title','Sex','Age']].head(10)
np.unique(train_df['Title'])
male_titles = ['Capt', 'Col', 'Don','Jonkheer','Major','Rev','Sir'] # all classified as male
female_titles = ['Countess','Dona','Lady'] # all as female
miss_titles = ['Mlle','Ms','Mme']

for df in [train_df, test_df]:
    df['Title'] = df['Title'].replace(miss_titles, 'Miss')
    df['Title'] = df['Title'].replace(female_titles, 'Mrs')
    df['Title'] = df['Title'].replace(male_titles, 'Mr')
    male_dr_filter = (df.Title == 'Dr') & (df.Sex == 'male')
    female_dr_filter = (df.Title == 'Dr') & (df.Sex == 'female')
    df.loc[male_dr_filter, ['Title']] = 'Mr'
    df.loc[female_dr_filter, ['Title']] = 'Mrs'
# let's quickly check how that worked for us
# combo_df has been created for quick data display
combo_df = pd.concat((train_df,test_df))
pd.crosstab(combo_df['Title'], combo_df['Sex'])
del combo_df
for df in [train_df, test_df]:
    df.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
train_df.head()
test_df.head()
# quickly check across both dataframes
combo_df = pd.concat((train_df,test_df))
combo_df.isnull().sum()
del combo_df
# Embarked - replace missing values (2 rows) with most common value (i.e. mode()) of Embarked field
for df in [train_df, test_df]:
    df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
for df in [train_df, test_df]:
    print(df['Embarked'].isnull().sum())
# Fare - replace the 0.0 values with mean fare by Pclass

# first replace all the Nan's with 0.0
values = {'Fare': 0.0} 
for df in [train_df, test_df]:
    df.fillna(value=values,inplace=True)
# create fare lookup tables for train_df & test_df
fare_table_train = (train_df.loc[train_df.Fare != 0.0][['Pclass','Fare']].groupby(['Pclass']).median()).copy()
fare_table_train
fare_table_test = (test_df.loc[test_df.Fare != 0.0][['Pclass','Fare']].groupby(['Pclass']).median()).copy()
fare_table_test
# here is how we can lookup the fare
for cls in np.unique(train_df.Pclass):
    print('%d %7.3f %7.3f' % (cls, fare_table_train.loc[cls], fare_table_test.loc[cls]))
train_df.loc[(train_df.Fare == 0.0)][:10] # view first 10 rows where Fare = 0.0
test_df.loc[(test_df.Fare == 0.0)][:10] # view first 10 rows where Fare = 0.0
for df, fare_table in zip([train_df, test_df],[fare_table_train, fare_table_test]):
    for cls in np.unique(df.Pclass):
        df.loc[((df.Fare == 0.0) & (df.Pclass == cls)), ['Fare']] = fare_table.loc[cls][0]
# let's check - we'll select some rows which had Fare == 0.0 above
train_df.loc[[180, 264, 272, 278]]
test_df.loc[[1044,1158,1264]]
# Age - replace age with mean age by title (which we had 'computed' earlier)

# find mean age by Title, for rows where age is not Null
age_lookup_train = (train_df.loc[(train_df['Age'] != 0.0), ['Age', 'Title']].groupby('Title').mean()).copy()
age_lookup_train
age_lookup_test = (test_df.loc[(test_df['Age'] != 0.0), ['Age', 'Title']].groupby('Title').mean()).copy()
age_lookup_test
train_df.loc[train_df.Age.isnull()][:10]
test_df.loc[test_df.Age.isnull()][:10]
for title in ['Master','Miss','Mr','Mrs']:
    print('%7s -> %8.3f %8.3f' % (title,age_lookup_train.loc[title], age_lookup_test.loc[title]))
# replace the Nulls from the Age lookups
for df, age_lookup in zip([train_df, test_df],[age_lookup_train, age_lookup_test]):
    for title in ['Master','Miss','Mr','Mrs']:
        df.loc[((df['Title'] == title) & (df['Age'].isnull())),['Age']] = age_lookup.loc[title][0]
# check if replacements were ok
train_df.loc[[6,20,29],['Title','Age']]
test_df.loc[[902,914,928],['Title','Age']]
# we will pick the following fields for running our models
fields_train = ['Survived', 'Pclass', 'Title', 'Age', 'Fare', 'Embarked','Fob']
fields_test = ['Pclass', 'Title', 'Age', 'Fare', 'Embarked','Fob']
# NOTE: 
# - dropping features Sex, SibSp, Parch from datasets
# - Sex not picked - Title picked instead, which includes 'Sex' implicitly
# - Sibsp and Parch not picked - Fob picked instead, which == 1 if SibSp > 0 or Parch > 0
train_df2 = train_df[fields_train]
test_df2 = test_df[fields_test]
train_df2.head()
test_df2.head()
# let's check if any of the features are co-related. We need to drop co-related cols
# I will consider cols significantly co-related if corr() >= 0.75

corr = train_df2.corr()
sns.heatmap(corr, cmap='BuPu', fmt='.2f',cbar=True, annot=True,annot_kws={"size": 10})
# Label encode the categorical features Title & Embarked
for df in [train_df2, test_df2]:
    df['Title'] = df['Title'].map( {'Miss': 4, 'Mrs': 3, 'Master': 2, 'Mr': 1} ).astype(int)
    df['Embarked'] = df['Embarked'].map( {'S': 3, 'C': 2, 'Q': 1} ).astype(int)
train_df2.head()
test_df2.head()
# Title & Embarked needs to be One-hot-encoded as well. We'll use pandas.get_dummies()
train_df2 = pd.get_dummies(train_df2,columns=['Title','Embarked'],drop_first=True)
test_df2 = pd.get_dummies(test_df2,columns=['Title','Embarked'],drop_first=True)
train_df2.head()
test_df2.head()
target_col = train_df2.columns.values[0]
columns = train_df2.columns.values[1:]
target_col, columns
# NOTE: we are going to be using k-fold cross validation, so it is not required to split the data into train/test sets
X_train, X_test = train_df2[columns], test_df2[columns]
y_train = train_df2[target_col]   # NOTE' there is no y_test!
X_train.shape, y_train.shape, X_test.shape
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
# Helper functions
def do_kfold_cv(classifier, X_train, y_train, n_splits=10, scoring='roc_auc'):
    """ do a k-fold cross validation run on classifier & training data
      and return cross-val scores """   
    kfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(classifier, X_train, y_train,scoring=scoring,cv=kfolds)
    return cv_scores

def test_classifier(clf_tuple, X_train, y_train, scoring='roc_auc', plot_feat_imp=False, verbose=2):
    """ run a k-fold test, fit model to training data & report scores for training & test data """
    # extract classifier instance & name
    clf, clf_name = clf_tuple
   
    if verbose > 0:
        print('Testing classifier %s...' % clf_name)
    
    clf.fit(X_train, y_train)
    
    acc_score = clf.score(X_train, y_train)

    # k-fold cross validation
    cv_scores = do_kfold_cv(clf, X_train, y_train)
    mean_cv_scores, std_cv_scores, min_cv_scores, max_cv_scores = \
        np.mean(cv_scores), np.std(cv_scores), np.min(cv_scores), np.max(cv_scores)
        
    # roc-auc score
    y_pred_proba = clf.predict_proba(X_train)[:,1]
    auc_score = roc_auc_score(y_train, y_pred_proba)

    if verbose > 1:   
        print('   - cross-val scores : Mean - %.3f Std - %.3f Min - %.3f Max - %.3f' % 
                  (mean_cv_scores, std_cv_scores, min_cv_scores, max_cv_scores))
        print('   - accuracy score   : %.3f' % acc_score)
        print('   - auc score        : %.3f' % auc_score)
        
    if plot_feat_imp:
        feat_imp = pd.DataFrame(clf.feature_importances_, index=columns, columns=['feature_imp'])
        feat_imp.sort_values(by='feature_imp', ascending=False, inplace=True)
        #print(feat_imp)
        feat_imp.plot(kind='bar', title='Feature Importances', legend=False)
        plt.ylabel('Feature Importance Score')
        plt.show()
        plt.close()

    return cv_scores, acc_score, auc_score

def test_classifiers(clf_list, X_train, y_train, scoring='roc_auc', verbose=2):
    """ run a list of classifiers against the training & test sets and
        return a pandas DataFrame of scores """
    classifier_names = []
    clf_cv_scores = []
    clf_acc_scores = []
    clf_auc_scores = []
    
    for clf_tuple in clf_list:
        cv_scores, acc_score, auc_score = \
            test_classifier(clf_tuple, X_train, y_train, scoring=scoring, verbose=verbose)
        classifier, classifier_name = clf_tuple
        classifier_names.append(classifier_name)
        clf_cv_scores.append(np.mean(cv_scores))
        clf_acc_scores.append(acc_score)
        clf_auc_scores.append(auc_score)
   
    # now create a DataFrame of all the scores & return
    scores_df = pd.DataFrame(data=clf_cv_scores, index=classifier_names,
                             columns=['cv_scores_mean'])
    scores_df['accuracy_scores'] = clf_acc_scores
    scores_df['auc_scores'] = clf_auc_scores
    return scores_df

def plot_roc_auc_curve(clf, X, y):
    clf.fit(X, y)
    preds = clf.predict_proba(X)[:,1]
    fpr, tpr, _ = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, lw=2, color='steelblue', label = 'AUC = %0.4f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],lw=2, color='firebrick',linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.close()
# now let's run a set of classifiers against the data
# we will use Pipelines() for all classifiers

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# create the classifiers to use...
clf_list = []

# KNN classifier
pipe_knn = Pipeline([('scl', StandardScaler()),
                     ('clf', KNeighborsClassifier(n_neighbors=5))])
clf_list.append((pipe_knn, 'KNN Classifier'))

# Logistic Regression classifier
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(penalty='l2', C=1.0, random_state=seed))])
clf_list.append((pipe_lr, 'LogisticRegression Classifier'))

# SVC (Linear) classifier
pipe_svcl = Pipeline([('scl', StandardScaler()),
                     ('clf', SVC(kernel='linear',C=1.0, gamma='auto', probability=True, random_state=seed))])
clf_list.append((pipe_svcl, 'SVC(Linear) Classifier'))

# SVC (Gaussian) classifier
pipe_svcg = Pipeline([('scl', StandardScaler()),
                     ('clf', SVC(kernel='rbf',C=1.0, gamma='auto', probability=True, random_state=seed))])
clf_list.append((pipe_svcg, 'SVC(Gaussian) Classifier')) 

# Naive Bayes
clf_nb = GaussianNB()
clf_list.append((clf_nb, 'Naive Bayes Classifier')) 

# DecisionTree classifier
clf_dt = DecisionTreeClassifier(max_depth=5, random_state=seed)
clf_list.append((clf_dt, 'Decision Tree Classifier'))

# ExtraTrees classifier
clf_xt = ExtraTreesClassifier(max_depth=5, n_estimators=100, random_state=seed)
clf_list.append((clf_xt, 'Extra Trees Classifier'))

# Random Forest classifier
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)
clf_list.append((clf_rf, 'Random Forests Classifier'))

# Gradient boosting classifier
clf_gbm = GradientBoostingClassifier(loss='deviance',learning_rate=0.1,
                                     n_estimators=100,max_depth=5,random_state=seed)
clf_list.append((clf_gbm, 'Gradient Boosting Classifier'))

# # Gradient boosting classifier
# clf_gbm2 = GradientBoostingClassifier(loss='deviance',learning_rate=0.1,
#                                      n_estimators=200,max_depth=5,random_state=seed)
# clf_list.append((clf_gbm2, 'Gradient Boosting Classifier-2'))
# test all the classifiers
scores_df = test_classifiers(clf_list, X_train, y_train, verbose=2)
print('Done!')
scores_df.sort_values(by=['accuracy_scores'], ascending=False, inplace=True)
print('\nClassifiers sorted by Accuracy Scores (descending):')
print(scores_df)

scores_df.sort_values(by=['cv_scores_mean'], ascending=False, inplace=True)
print('\nClassifiers sorted by mean cv_scores (descending):')
print(scores_df)

scores_df.sort_values(by=['auc_scores'], ascending=False, inplace=True)
print('\nClassifiers sorted by AUC scores (descending):')
print(scores_df)
# plot the ROC-AUC curve for the 'best classifier'
plot_roc_auc_curve(clf_gbm, X_train, y_train)
from sklearn.model_selection import learning_curve, validation_curve

def get_lc_scores(clf, X_train, y_train, train_size_props=None, cv_folds=10, verbose=False):
    # calculate scores for learning curves
    #train_size_props = [0.10, 0.25, 0.33, 0.45, 0.50, 0.66, 0.75, 0.85, 0.90, 1.0]
    if train_size_props is None:
        train_size_props = np.linspace(0.1,1.0,10)
    train_sizes, train_scores, test_scores = \
      learning_curve(clf, X_train, y_train,train_sizes=train_size_props,cv=cv_folds)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    if verbose:
        print('Train sizes:') 
        print(train_sizes)
        print('\nTraining mean scores')
        print(train_mean)
        print('\nValidation mean scores')
        print(test_mean)
    return train_sizes, train_mean, train_std, test_mean, test_std

def plot_learning_curve(clf, X_train, y_train, desired_accuracy_score,train_size_props=None, cv_folds=10, verbose=False):
    train_sizes, train_mean, train_std, test_mean, test_std = \
        get_lc_scores(clf, X_train, y_train, train_size_props, cv_folds, verbose)
    plt.subplots(figsize=(6,4))
    plt.plot(train_sizes, train_mean, lw=2, color='steelblue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='steelblue')
    plt.plot(train_sizes, test_mean, lw=2, color='forestgreen', marker='o', markersize=5, label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='forestgreen')
    plt.axhline(y=desired_accuracy_score, lw=2, color='firebrick', linestyle='--', label='Desired Accuracy')
    plt.xlabel('No. of training samples')
    plt.ylabel('Accuracy Scores')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.show()
    plt.close()
# picking out the desired accuracy from scores_df calculated above
scores_df.sort_values(by=['accuracy_scores'], ascending=False, inplace=True)
desired_accuracy_score = scores_df.iloc[0]['accuracy_scores']
desired_accuracy_score
plot_learning_curve(clf_gbm, X_train, y_train, desired_accuracy_score)
from sklearn.decomposition import PCA

clf_list2 = []

clf_list2.append((clf_gbm, 'Gradient Boosting Classifier [PCA=All]'))

clf_gbm5 = Pipeline([('pca', PCA(n_components=5)),
                     ('clf', GradientBoostingClassifier(loss='deviance',learning_rate=0.1,
                                                        n_estimators=100,max_depth=5,random_state=seed))])
clf_list2.append((clf_gbm5, 'Gradient Boosting Classifier [PCA=5]'))

clf_gbm4 = Pipeline([('pca', PCA(n_components=4)),
                     ('clf', GradientBoostingClassifier(loss='deviance',learning_rate=0.1,
                                                        n_estimators=100,max_depth=5,random_state=seed))])
clf_list2.append((clf_gbm4, 'Gradient Boosting Classifier [PCA=4]'))

clf_gbm3 = Pipeline([('pca', PCA(n_components=3)),
                     ('clf', GradientBoostingClassifier(loss='deviance',learning_rate=0.1,
                                                        n_estimators=100,max_depth=5,random_state=seed))])
clf_list2.append((clf_gbm3, 'Gradient Boosting Classifier [PCA=3]'))
scores_df2 = test_classifiers(clf_list2, X_train, y_train, verbose=2)
print('Done!')
scores_df2.sort_values(by=['accuracy_scores'], ascending=False, inplace=True)
print('\nClassifiers sorted by Accuracy Scores (descending):')
print(scores_df2)

scores_df2.sort_values(by=['cv_scores_mean'], ascending=False, inplace=True)
print('\nClassifiers sorted by mean cv_scores (descending):')
print(scores_df2)

scores_df2.sort_values(by=['auc_scores'], ascending=False, inplace=True)
print('\nClassifiers sorted by AUC scores (descending):')
print(scores_df2)
# now let's predict the results on the test dataset
y_pred = clf_gbm.predict(X_test)  # GBC with all features
test_df2['Survived-A'] = y_pred

y_pred5 = clf_gbm5.predict(X_test)  # GBC with PCA=5
test_df2['Survived'] = y_pred5

test_df2.head()
out_file = './gender_submission.csv'
test_df2.to_csv(out_file,columns=['Survived'])
%pprint
open(out_file).readlines()[:10]
