# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np                # linear algebra

import pandas as pd               # data processing, CSV file I/O (e.g. pd.read_csv)

#import fancyimpute as fi         # MICE completions

import re                         # Regular expressions

import matplotlib.pyplot as plt   # Plot various stuff

import sklearn                    # Learning algorithm

import sklearn.tree               # Decision trees

import sklearn.preprocessing      # Preprocessors

import sklearn.neighbors          # k-Nearest Neighbours

import sklearn.naive_bayes        # Naive Bayes

import sklearn.svm                # C-Support SVM

import sklearn.linear_model       # Logistic regression

import mlxtend.classifier         # Stacking Classifier

import sklearn.feature_extraction # Extract features through DictVectorizer

import sklearn.ensemble           # Random Forest

import sklearn.feature_selection  # SelectFromModel

import sklearn.ensemble.gradient_boosting

# Load the files



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



# The columns are

# PassengerID  : The passenger's ID

# Survived     : 1 if the passenger survived, 0 otherwise

# Pclass       : Passenger class, which can be considered a proxy for social status

# Name         : Passenger's name

# Sex          : Passenger's sex

# Age          : Passenger's age

# SibSp        : Number of siblings or spouses aboard

# Parch        : Number of parents or children aboard

# Ticket       : Ticket number

# Fare         : Fare paid, another variable to indicate social status

# Cabin        : Denomination of the cabin

# Embarked     : Port of embarkation



# Remove the target

targets = train.Survived

testid = test.PassengerId

train.drop('Survived',1,inplace=True)



# Combine the datasets

combined = train.append(test)

combined.reset_index(inplace=True)

combined.drop('index',inplace=True,axis=1)



# Some of the data is missing, for example



# 10	1	2	Nasser, Mrs. Nicholas (Adele Achem)	female	14	1	0	237736	30.0708		C

# Which lacks the cabin.



# Which data is complete and useful

# PassengerID, unlikely to be useful (this is a unique ID), but needed to uniquely identify a passenger

# Survived, which is what we need to predict

# Pclass, very useful (Survival rate 1st class is +/-63%, 3rd class is about 24%)

# Name, could be useful for the last name as it will group the families together

# Sex, very useful as this was still a time when "children and ladies first" was applied during naufrage

# Age, would be useful but a 177 rows are missing. Maybe we can reconstruct this.

# SibSp, useful: it shows that single people (SibSp==0) had less chance of survival than couples 

#                or parent-children (53%), but then that the higher the SibSp, the lower the survival rate

# Parch, useful, similar behaviour as SibSp

# Ticket, could be useful, but need some cleanup. Normally numeric but some have characters attached

# Fare, likely to be useful

# Cabin, could have been useful, but there are too many values missing in the training set. Instead the

# Pclass will be used.

# Embarked, useful.



# Augmentation of the dataset

# From the data provided we will derive

# LastName   - the element before the "," in the Name field

# TicketNum  - the numeric part of the ticket



# Prepare the dataset

# Separate the title from the rest, children were often called "Miss" or "Master" at the time

title_re = re.compile("^.[^,]+, ([^.]+)\. .*$")

combined['Title'] = combined['Name'].map(lambda x: title_re.search(x).group(1))

# These titles contain an information about the age and the sex, so I will create a label

# gtitle that contains a general description. Also, some labels are in non-english languages.

gtitle_dict = {

    'Mr': 'Civil',

    'Mrs': 'Civil',

    'Miss': 'Civil',

    'Master': 'Civil',

    'Don': 'Nobility',

    'Rev': 'Officer',

    'Dr': 'Civil',

    'Mme': 'Civil',

    'Ms': 'Civil',

    'Major': 'Officer',

    'Lady': 'Nobility',

    'Sir': 'Nobility',

    'Mlle': 'Civil',

    'Col': 'Officer',

    'Capt': 'Officer',

    'the Countess': 'Nobility',

    'Jonkheer': 'Nobility',

    'Dona': 'Nobility'

}

combined['gtitle'] = combined.Title.map(gtitle_dict)



# Check if we have missing data

print("=== BEFORE IMPUTERS ===\n")

print("=== Check for missing values in set ===")

for iname in combined.columns.values.tolist():

    nna = sum(pd.isnull(combined[iname]))

    print(repr(iname).rjust(16), repr(nna).rjust(4))



# Complete / impute the datasets



# Fill the missing ages

# For the missing ages, we will consider that the following variables are representative of the age:

# Pclass, Sex, Title

# Name - unlikely to be helpful as this is likely to be almost unique

# LastName - unlikely to be helpful

# SibSp/Parch - could be useful, but need a lot of work

# Ticket - unlikely to be useful, likely to be unique

# Fare - unlikely to be useful

# Cabin - could have been useful if there wasn't so many missing values

# Embarked - could have been useful, but this chops the dataset into too many fragments

# For each useful variable, I will iterate through the possible values (They are all categoricals)

# and if there are any missing value:

# Find the min, the max and select a value at random between these.

# Now comes the difficult choice of what distribution to use

# Uniform? T-Student? Gamma? Something else?

# Major drawback - as there is an element of randomness, the behavior can change from run to run

# Some data

# In the training set

# Pclass  Title   Sex   Number NA  Number Entries  Percentage NA  Age Min  Age Max  Age Median

#      3  Mr.     male         90             229          39.30       11       74          26

#      3  Mrs.    female        9              33          27.27       15       63          31

#      3  Miss.   female       33              69          47.83        0       45          18

#      3  Master. male          4              24          16.67        0       12           4

#      1  Mr.     male         20              87          22.99       17       80          40

#      1  Mrs.    female        8              34          23.53       17       62          41.5

#      1  Miss.   female        1              45           2.22        2       63          30

#      1  Dr.     male          1               3          33.33       32       50          44

#      2  Mr.     male          9              82          10.98       16       70          31

#      2  Miss.   female        2              32           6.25        2       50          24

# The most problematic imputation will be 3rd Class/Female/Miss which have almost half of 

# the entries missing the age. It seems the term "Miss" was used in a variety of circumstances across

# the three classes. Was that intentional or by mistake?

def age_imputer(df_to_imp, set_name, doplot=False):

    for a1 in df_to_imp['Pclass'].unique():

        for a2 in df_to_imp['Title'].unique():

            for a3 in df_to_imp['Sex'].unique():

                #query = {'Pclass': [a1],

                #         'Sex': [a3],

                #         'Title': [a2]}

                #mask = train[['Pclass','Title','Sex']].isin(query).all(1)

                v1 = df_to_imp[(df_to_imp.Pclass==a1)&(df_to_imp.Title==a2)&(df_to_imp.Sex==a3)]['Age'].count()

                v2 = sum(np.isnan(df_to_imp[(df_to_imp.Pclass==a1)&(df_to_imp.Title==a2)&\

                                            (df_to_imp.Sex==a3)]['Age']))

                # If we have at least one NA and at least one non-NA value

                if (v2 > 0) and (v1 > v2):

                    age_dist = np.array(df_to_imp[(df_to_imp.Pclass==a1)&(df_to_imp.Title==a2)&\

                                                  (df_to_imp.Sex==a3)]['Age'])

                    age_dist = age_dist[np.logical_not(np.isnan(age_dist))]

                    age_dist_before = age_dist.copy()

                    # To make it simple - I will select as many elements as I have NAs to replace

                    # in the list of existing values with respect to distribution. At least the distribution

                    # is respected

                    # Replacing the random selection with the median for all values does not

                    # significantly change the results

                    new_age = np.random.choice(age_dist,v2,replace=True)

                    #new_age = np.median(age_dist)

                    # Impute the missing values

                    df_to_imp.loc[(df_to_imp.Pclass==a1)&(df_to_imp.Title==a2)&(df_to_imp.Sex==a3)&\

                                  (pd.isnull(df_to_imp.Age)),'Age'] = new_age

                    if doplot:

                        # Plot distributions (old and new)

                        plt.subplot(2,1,1)

                        plt.title(("Distribution for Pclass: {0:d}, Sex: {1}, Title: {2}\n" +\

                                   "(imputed: {3:d} value{4})\n(Set name: {5})").format(

                                  a1, a3, a2, v2, 's' if (v2>1) else '', set_name))

                        plt.hist(age_dist_before, normed=True)

                        plt.subplot(2,1,2)

                        plt.hist(df_to_imp[(df_to_imp.Pclass==a1)&(df_to_imp.Title==a2)&\

                                           (df_to_imp.Sex==a3)]['Age'], normed=True)

                        plt.show()



age_imputer(combined, "Combined dataset", doplot=False)



print("=== AFTER IMPUTERS ===\n")

print("=== Check for missing values in set ===")

for iname in combined.columns.values.tolist():

    nna = sum(pd.isnull(combined[iname]))

    print(repr(iname).rjust(16), repr(nna).rjust(4))

    

# At this point, there is still a row with age == NULL

# She is the only "Ms." "female" "3rd Pclass"

# Given that she is a single lady (Parch==SibSp==0),  let's assign her

# The median age of the single ladies after the coming of age



combined.loc[(combined.PassengerId==980), 'Age'] = np.median(combined.loc[(combined.Pclass==3)&\

                                                                          (combined.Sex=='female')&\

                                                                          (combined.Parch==0)&\

                                                                          (combined.SibSp==0)&\

                                                                          (combined.Age>17),'Age'])



# At this point, we are done with completing Age. Let's take care of the fare.

# Fare is pretty easy - only one value is missing in the test dataset.

# He is in 3rd Pclass, left from Southampton.

# Let's assign him the median of the tickets of childless single men who embarked in Southampton

# in third class over 40 years old.



combined.loc[(combined.PassengerId==1044),'Fare'] = combined.loc[(combined.Pclass==3)&\

                                                                 (combined.Embarked=='S')&\

                                                                 (combined.Parch==0)&\

                                                                 (combined.SibSp==0)&\

                                                                 (combined.Age>40)&\

                                                                 -(pd.isnull(combined.Fare)),'Fare'].median()



# Some of the entries have a Fare of 0.0. I do not know if this is normal or not, and in doubt

# I will leave them that way. One possible explanation is they were part of the personnel 

# of either White Star Line or Harland and Wolff.



# Lastly, I will not fix the Embarked variable as this does not provide any information

# regarding the survival rate.



# Well, that is it. The datasets are relatively complete, imputed where a value was missing 

# so we are ready to rock.



# Let's create a feature based on Parch and SibSp

# Note: if the grand-parents, parents and children are on the Titanic ... I do 

# not know how that works.

# It is also possible that two siblings with no parents are reported as a couple.



def class_family(Parch, SibSp):

    if Parch == 0:

        # No parent nor children

        if SibSp == 0:

            # Traveling without parents/children and without spouse/siblings

            return "Single"

        elif SibSp == 1:

            # Traveling with either a spouse or with a sibling but no children or parent

            return "Couple"

        else:

            # Traveling with no parent/children but several Sibling or (...) spouses

            return "Family"

    if Parch == 1:

        # That is either one parent or one children

        # There we have SibSp taking values 0, 2, 3 or 4

        # 0 is likely to be a children traveling with his/her mother or father

        # In all cases, Families

        return "Family"

    # All other cases are families

    return "Family"



combined["Type"] = combined[["Parch", "SibSp"]].apply(lambda x: class_family(x["Parch"],

                                                                             x["SibSp"]),

                                                                             axis=1)

# Let's sort the units per size (Small, Medium, Large)

# Small is 1 to 3 members

# Medium is 4 to 6

# Large is anything above 6

def size_family(nmembers):

    if nmembers < 4:

        return "Small"

    elif nmembers < 7:

        return "Medium"

    else:

        return "Large"



combined["Size"] = combined[["Parch", "SibSp"]].apply(lambda x: size_family(x["Parch"]+

                                                                            x["SibSp"]+1),

                                                                            axis=1)

combined["NMembers"] = combined[["Parch", "SibSp"]].apply(lambda x: x["Parch"]+

                                                                    x["SibSp"]+1,

                                                                    axis=1)

# Fix the ticket by filling with "X" if no value

combined.Cabin.fillna('X',inplace=True)

combined.Cabin = combined.Cabin.map(lambda x: x[0])

# Let's drop the unneeded labels

combined.drop(['Name','Embarked','Ticket',

               'Parch','SibSp', 'Title',

               'PassengerId'],1, inplace=True)



# We have several variables to transform to 0/1 features

# Let's use a simple dummy encoder

# Index(['PassengerId', 'Pclass', 'Sex', 'Age', 'Fare', 'Cabin', 'gtitle',

#       'Type', 'Size'],



# PClass

d_pclass = pd.get_dummies(combined['Pclass'],prefix='PClass')

d_gtitle = pd.get_dummies(combined['gtitle'],prefix='GTitle')

d_type = pd.get_dummies(combined['Type'],prefix='FType')

d_size = pd.get_dummies(combined['Size'],prefix='FSize')

d_cabin = pd.get_dummies(combined['Cabin'], prefix="Cabin")

combined = pd.concat([combined,d_pclass, d_gtitle, d_type, d_size, d_cabin],axis=1)

# Map Sex to 0 (male) / 1 (female)

combined['Sex'] = combined['Sex'].map({'male':0,'female':1})

# And scale the Age and fare so the max is 1

combined.Fare = combined.Fare / combined.Fare.max()

combined.Age = combined.Age / combined.Age.max()

combined.NMembers = combined.NMembers / combined.NMembers.max()

# And drop the unused variables

combined.drop(['Pclass','gtitle','Type','Size','Cabin'],axis=1,inplace=True)

# How many variables?

print("\n\nNumber of variables: %d"%(len(combined.columns)))

# 16 Columns, still playable

# Recreate the data sets we need for the training

train_ext = combined.ix[0:890]

test_ext = combined.ix[891:]



# Let's have a peek at the importance of each feature

clf = sklearn.ensemble.ExtraTreesClassifier(n_estimators=200)

clf = clf.fit(train_ext, targets)

features = pd.DataFrame()

features['feature'] = train_ext.columns

features['importance'] = clf.feature_importances_

print("\nFeatures and importance")

print(features.sort_values(by=['importance'],ascending=False))



def best_params(clf, clf_name, clf_grid):

    global train_ext

    global targets

    cv = sklearn.model_selection.StratifiedKFold(n_splits=5)

    gs = sklearn.model_selection.GridSearchCV(clf,

                                              param_grid=clf_grid,

                                              cv=cv)

    gs.fit(train_ext, targets)

    print("\n%s"%(clf_name))

    print("="*len(clf_name))

    print('Best score: {}'.format(gs.best_score_))

    print('Best parameters: {}'.format(gs.best_params_))

    return gs.best_params_

   

# Search over a grid for optimal KNN parameters

clf_knn = sklearn.neighbors.KNeighborsClassifier()

parameter_grid = {'weights': ['uniform','distance'],

                  'n_neighbors': [3,4,5,6,7,8,9]}

knn_bp = best_params(clf_knn, "KNN", parameter_grid)

# Search over a grid for optimal decision tree parameters

clf_dt = sklearn.tree.DecisionTreeClassifier()

parameter_grid = {'criterion': ['gini','entropy'],

                  'splitter': ['best','random'],

                  'max_depth': [4,5,6,7,8,9,10],

                  'max_features': ['sqrt',0.5, None]}

dt_bp = best_params(clf_dt, "Decision Tree", parameter_grid)

# Search over a grid for optimal SVC parameters

clf_svc = sklearn.svm.SVC()

parameter_grid = {'kernel': ['linear','poly', 'rbf','sigmoid'],

                  'degree': [2,3,4,5]}

svc_bp = best_params(clf_svc, "C-Support Vector classification", parameter_grid)



# Search over a grid for optimal random forest parameters

clf_rf = sklearn.ensemble.RandomForestClassifier()

parameter_grid = {'max_features': ['sqrt',0.5, None],

                  'max_depth' : [6,7,8,9],

                  'n_estimators': [200,210,240,250],

                  'criterion': ['gini','entropy']

                 }

rf_bp = best_params(clf_rf, "Random Forest", parameter_grid)

# I got the best parameters for each classifier

# Let's create each classifier and use a stack

clf_knn = sklearn.neighbors.KNeighborsClassifier(**knn_bp)

clf_dt = sklearn.tree.DecisionTreeClassifier(**dt_bp)

clf_svc = sklearn.svm.SVC(**svc_bp)

clf_rf = sklearn.ensemble.RandomForestClassifier(**rf_bp)

# Build the stacked classifier

lr = sklearn.linear_model.LogisticRegression()

clf_st = mlxtend.classifier.StackingClassifier(classifiers=[clf_knn, clf_dt, clf_svc, clf_rf],

                            meta_classifier=lr)

blah = best_params(clf_st,"Stacked Classifier", {})



clf_st.fit(train_ext,targets)

test_pred = clf_st.predict(test_ext)



submission = pd.DataFrame({ 'PassengerId': testid,

                            'Survived': test_pred})

submission.to_csv("titanic_prediction.csv", header=True, index=False)