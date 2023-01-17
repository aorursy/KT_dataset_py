# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from IPython.display import display



# remove warnings

import warnings

warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.

# Loading Modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() # setting seaborn default for plots



# Loading Datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Data Prepration

# save PassengerId for final submission

passengerId = test.PassengerId

# merge train and test

titanic = train.append(test, ignore_index=True)





# create indexes to separate data later on

train_idx = len(train)

test_idx = len(titanic) - len(test)
# Let's take a look

# describe(include = ['O']) will show the descriptive statistics of object data types

titanic.describe(include=['O'])
display( titanic.head(5), 

titanic.info(), 

titanic.isnull().sum()

       )
# Survival and Precent

survived = titanic[titanic['Survived'] == 1]

not_survived = train[train['Survived'] == 0]             # Using train dataset since titanic includes test as well



print ("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(train)*100.0))

print ("Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train)*100.0))

print ("Total: %i"%len(train))

# Mathmetically 

print( train.groupby('Pclass').Survived.value_counts(), 

      train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean()

     )



sns.barplot(x='Pclass', y='Survived', data=train)
print( titanic.Sex.value_counts(),

train.groupby('Sex').Survived.value_counts())



sns.barplot(x='Sex', y='Survived', data=train)
tab = pd.crosstab(titanic['Pclass'], titanic['Sex'])

print (tab)



# Floating division of dataframe and other, element-wise (binary operator truediv).

# Equivalent to dataframe / other, but with support to substitute a fill_value for missing data in one of the inputs. With reverse version, rtruediv.

tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)



plt.xlabel('Pclass')

plt.ylabel('Percentage')

plt.xticks(rotation=0)



# sns.factorplot('Sex', 'Survived', hue='Pclass', size=4, aspect=2, data=train)

sns.catplot('Sex', 'Survived', hue='Pclass', height=4, aspect=2, data=train, kind='point')
sns.catplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=train, kind = "point" )
print( train.Embarked.value_counts(), 

     train.groupby('Embarked').Survived.value_counts(),

     train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())



sns.barplot('Embarked', 'Survived', hue='Sex', data = train)
print( train.Parch.value_counts(),

train.groupby('Parch').Survived.value_counts(),

train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean() )



sns.barplot(x='Parch', y='Survived', ci=None, data=train) # ci=None will hide the error bar
print( train.SibSp.value_counts(), 

      train.groupby('SibSp').Survived.value_counts(), 

      train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()

)



sns.barplot( x = 'SibSp', y = 'Survived', ci = None, data = train)
fig , (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

fig.set_size_inches(15,5)



sns.violinplot(x="Embarked", y="Age", hue="Survived", data=train, split=True, ax=ax1)

sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train, split=True, ax=ax2)

sns.violinplot(x="Sex", y="Age", hue="Survived", data=train, split=True, ax=ax3)
total_survived = train[train['Survived']==1]

total_not_survived = train[train['Survived']==0]

male_survived = train[(train['Survived']==1) & (train['Sex']=="male")]

female_survived = train[(train['Survived']==1) & (train['Sex']=="female")]

male_not_survived = train[(train['Survived']==0) & (train['Sex']=="male")]

female_not_survived = train[(train['Survived']==0) & (train['Sex']=="female")]



import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(2, 2, right=2, top= 2)

ax1 = plt.subplot(gs[0, :])

ax2 = plt.subplot(gs[1,0])

ax3 = plt.subplot(gs[1,1])



sns.distplot(total_survived['Age'].dropna().values, bins=range(0, 81, 2), kde=False, color='blue', ax = ax1)

sns.distplot(total_not_survived['Age'].dropna().values, bins=range(0, 81, 2), kde=False, color='red', axlabel='Age', ax = ax1)



sns.distplot(female_survived['Age'].dropna().values, bins=range(0, 81, 2), kde=False, color='blue', ax = ax2)

sns.distplot(female_not_survived['Age'].dropna().values, bins=range(0, 81, 2), kde=False, color='red', axlabel='Female Age', ax = ax2)



sns.distplot(male_survived['Age'].dropna().values, bins=range(0, 81, 2), kde=False, color='blue', ax = ax3)

sns.distplot(male_not_survived['Age'].dropna().values, bins=range(0, 81, 2), kde=False, color='red', axlabel='Male Age', ax = ax3)



ax1.legend(['Total Survived', 'Total Not Survived'],loc="upper right")

ax2.legend(['Females Survived', 'Females Not Survived'],loc="upper right")

ax3.legend(['Males Survived', 'Males Not Survived'],loc="upper right")

plt.figure(figsize=(15,6))

sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=0.7, square=False, annot=True)
# Extracting Title 

# create a new feature to extract title names from the Name column

titanic['Title'] = titanic.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())



# normalize the titles

normalized_titles = {

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royalty"

}

# map the normalized titles to the current titles 

titanic.Title = titanic.Title.map(normalized_titles)

# view value counts for the normalized titles

pd.crosstab(titanic['Title'], titanic['Sex'])

print(titanic.Title.value_counts())
# A Scikit Label Encoder 

from sklearn.preprocessing import LabelEncoder



# Encoding Titles

title_encoder = LabelEncoder()



# Fitting

titanic['Title'] = title_encoder.fit_transform(titanic['Title'])

# test['Title'] = title_encoder.transform( test['Title'] )



# Viewing results

print( titanic.head(),

titanic.info() )
titanic.info()

titanic.Cabin[:10]
# Fill Cabin NaN with U for unknown

titanic.Cabin = titanic.Cabin.fillna('U')



# map first letter of cabin to itself

titanic['Deck'] = titanic.Cabin.map(lambda x: x[0])



# Create room number

titanic['Room'] = titanic['Cabin'].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")





### Filling missing data





#     Now we will fill the missing data

# titanic["Deck"] = titanic["Deck"].fillna("U")

titanic["Room"] = titanic["Room"].fillna(titanic["Room"].mean())

titanic.head()
# Encoding Gender

# Should investigate further setting labels for LE, currently alphabatic



sex_encoder = LabelEncoder()

X = ['male', 'female']



# Fitting

sex_encoder.fit(X)



# Applying

titanic['Sex'] = sex_encoder.fit_transform(titanic['Sex'])

# test['Sex'] = sex_encoder.transform(test['Sex'])



# display( train.head(),

#        test.head())



# print( titanic.Sex.unique())
display( titanic.Embarked.unique(),



# Checking the numbers

titanic.Embarked.value_counts(),

titanic.Embarked.isna().sum() 

       )
# Adding the value of the most common Embarkation point



# find most frequent Embarked value and store in variable

most_embarked = titanic.Embarked.value_counts().index[0]

# fill NaN with most_embarked value

titanic.Embarked = titanic.Embarked.fillna(most_embarked)
titanic.info()
# for dataset in train_test_data:

#     #print(dataset.Embarked.unique())

#     dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



titanic['Embarked'] = titanic['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

titanic['Embarked'].head()
# group by Sex, Pclass, and Title 

grouped = titanic.groupby(['Sex','Pclass', 'Title'])  

# view the median Age by the grouped features 

grouped.Age.median()

# apply the grouped median value on the Age NaN

titanic.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))



# for dataset in train_test_data:

#     age_avg = dataset['Age'].mean()

#     age_std = dataset['Age'].std()

#     age_null_count = dataset['Age'].isnull().sum()

#     age_null_random_list = np.random.randint ( age_avg - age_std, age_avg + age_std, size = age_null_count)

#     dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

#     dataset['Age'] = dataset['Age'].astype(int)

    



# Transforming into bands 

# train['AgeBand'] = pd.cut(train['Age'], 5)

# print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())

# # train.drop('AgeBand', axis = 1)



# # Applying the change to Age

# for dataset in train_test_data:

#     dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

#     dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

#     dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

#     dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

#     dataset.loc[ dataset['Age'] > 64, 'Age'] = 4



# Viewing 

titanic.head()

titanic.info()
titanic.head()
# group by Sex, Pclass, and Title 

fare_group = titanic.groupby(['Pclass', 'Deck', 'Room'])  



# view the median Age by the grouped features 

fare_group.Fare.median()

# apply the grouped median value on the Age NaN

titanic.Fare = fare_group.Fare.apply(lambda x: x.fillna(x.median()))



# # FareBand

# titanic['FareBand'] = pd.qcut(titanic['Fare'], 4)

# print (titanic[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())

# print( pd.qcut(titanic['Fare'], 4).unique())



# # Mapping to Band

# # for dataset in train_test_data:

# #     dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

# #     dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

# #     dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

# #     dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

# #     dataset['Fare'] = dataset['Fare'].astype(int)

    

display(titanic.head(), 

     grouped.Age.apply(lambda x: x.fillna(x.median())),

titanic.info())
# for dataset in train_test_data:

#     dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1           # Adding 1 to count for the person



# print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())



# size of families (including the passenger)

titanic['FamilySize'] = titanic.Parch + titanic.SibSp + 1



print(titanic.FamilySize[:5])
# for dataset in train_test_data:

#     dataset['IsAlone'] = 0

#     dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

# print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())



titanic['IsAlone'] = 0

titanic.loc[titanic['FamilySize'] == 1, 'IsAlone'] = 1

print (titanic[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

# create train and test data

train = titanic[ :train_idx]

test = titanic[test_idx: ]

# convert Survived back to int

train.Survived = train.Survived.astype(int)



# create X and y for data and target values 

X_train = train.drop('Survived', axis=1)

y_train = train.Survived

# create array for test set

X_test = test.drop('Survived', axis=1)





display( X_train[:5], y_train[:5], X_test[:5] )

X_train.shape, y_train.shape, X_test.shape
features = [ 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title','SibSp',

       'Parch', 'FamilySize', 'IsAlone']



# features = [ 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title',

#         'FamilySize', 'IsAlone']





X_train = X_train[features]

# y_train = train.Survived

# create array for test set

X_test = X_test[features]
display(

titanic.isna().sum())
# Importing Classifier Modules

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_score

from sklearn import model_selection
clf = LogisticRegression()

clf.fit(X_train, y_train)

y_pred_log_reg = clf.predict(X_test)



cv_scores = cross_val_score(clf, X_train, y_train, scoring='accuracy')*100

print('Model Accuracy %.2f' %(cv_scores.mean()*100))



# Create MLA Table

Algorithms = pd.DataFrame( columns = ['ML Algorithm', 'Parameters','Train Accuracy Mean', 'Test Accuracy Mean', 'Mean Absolute Error', 'Test Accuracy 3*STD' ,'Time', 'Output'])



## split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit

#note: this is an alternative to train_test_split

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%



# CV Results

cv_results = model_selection.cross_validate(clf, X_train, y_train, cv  = cv_split)

# Appending to Algorithms evaluation DF

Algorithms =  Algorithms.append( {'ML Algorithm' : clf.__class__.__name__ , 

                                     'Parameters': clf.get_params(),

                                     'Train Accuracy Mean' : cv_results['train_score'].mean(), 

                                     'Test Accuracy Mean' : cv_results['test_score'].mean(), 

                                      'Mean Absolute Error' : cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error').mean() * -1 ,

                                      'Test Accuracy 3*STD' : cv_results['test_score'].std()*3,

                                      'Time' : cv_results['fit_time'].mean(),

                                     'Output' : 'y_pred_log_reg'}, ignore_index=1)

display(Algorithms)
clf = SVC()

clf.fit(X_train, y_train)

y_pred_svc = clf.predict(X_test)

acc_svc = round(clf.score(X_train, y_train) * 100, 2)

print (acc_svc)



cv_scores = cross_val_score(clf, X_train, y_train, scoring='accuracy')

print('Model Accuracy %.2f' %(cv_scores.mean()*100))



scores = cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error')

print('Mean Absolute Error %2f' %(-1 * scores.mean()))

mae_svc = -1 * scores.mean()





# CV Results

cv_results = model_selection.cross_validate(clf, X_train, y_train, cv  = cv_split)

# Appending to Algorithms evaluation DF

Algorithms =  Algorithms.append( {'ML Algorithm' : clf.__class__.__name__ , 

                                     'Parameters': clf.get_params(),

                                     'Train Accuracy Mean' : cv_results['train_score'].mean(), 

                                     'Test Accuracy Mean' : cv_results['test_score'].mean(), 

                                  'Mean Absolute Error' : cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error').mean() * -1 ,

                                     'Test Accuracy 3*STD' : cv_results['test_score'].std()*3,

                                      'Time' : cv_results['fit_time'].mean(),

                                     'Output' : 'y_pred_svc'}, ignore_index=1)
clf = LinearSVC()

clf.fit(X_train, y_train)

y_pred_linear_svc = clf.predict(X_test)

acc_linear_svc = round(clf.score(X_train, y_train) * 100, 2)

print (acc_linear_svc)



cv_scores = cross_val_score(clf, X_train, y_train, scoring='accuracy')

print('Model Accuracy %.2f' %(cv_scores.mean()*100))



scores = cross_val_score(clf, train[features], train['Survived'], scoring='neg_mean_absolute_error')

print('Mean Absolute Error %2f' %(-1 * scores.mean()))

mae_lsvc = -1 * scores.mean()





# CV Results

cv_results = model_selection.cross_validate(clf, X_train, y_train, cv  = cv_split)

# Appending to Algorithms evaluation DF

Algorithms =  Algorithms.append( {'ML Algorithm' : clf.__class__.__name__ , 

                                     'Parameters': clf.get_params(),

                                     'Train Accuracy Mean' : cv_results['train_score'].mean(), 

                                     'Test Accuracy Mean' : cv_results['test_score'].mean(), 

                                  'Mean Absolute Error' : cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error').mean() * -1 ,

                                     'Test Accuracy 3*STD' : cv_results['test_score'].std()*3,

                                      'Time' : cv_results['fit_time'].mean(),

                                     'Output' : 'y_pred_linear_svc'}, ignore_index=1)
clf = KNeighborsClassifier(n_neighbors = 10)

clf.fit(X_train, y_train)

y_pred_knn = clf.predict(X_test)

acc_knn = round(clf.score(X_train, y_train) * 100, 2)

print (acc_knn)

scores = cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error')

print('Mean Absolute Error %2f' %(-1 * scores.mean()))

mae_knn = -1 * scores.mean()





# CV Results

cv_results = model_selection.cross_validate(clf, X_train, y_train, cv  = cv_split)

# Appending to Algorithms evaluation DF

Algorithms =  Algorithms.append( {'ML Algorithm' : clf.__class__.__name__ , 

                                     'Parameters': clf.get_params(),

                                     'Train Accuracy Mean' : cv_results['train_score'].mean(), 

                                     'Test Accuracy Mean' : cv_results['test_score'].mean(), 

                                  'Mean Absolute Error' : cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error').mean() * -1 ,

                                     'Test Accuracy 3*STD' : cv_results['test_score'].std()*3,

                                      'Time' : cv_results['fit_time'].mean(),

                                     'Output' : 'y_pred_knn'}, ignore_index=1)
clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)



y_pred_decision_tree = clf.predict(X_test)

acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)

print (acc_decision_tree)

scores = cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error')

print('Mean Absolute Error %2f' %(-1 * scores.mean()))

mae_decision_tree = -1 * scores.mean()





# CV Results

cv_results = model_selection.cross_validate(clf, X_train, y_train, cv  = cv_split)



# Appending to Algorithms evaluation DF

Algorithms =  Algorithms.append( {'ML Algorithm' : clf.__class__.__name__ , 

                                     'Parameters': clf.get_params(),

                                     'Train Accuracy Mean' : cv_results['train_score'].mean(), 

                                     'Test Accuracy Mean' : cv_results['test_score'].mean(), 

                                     'Test Accuracy 3*STD' : cv_results['test_score'].std()*3,

                                  'Mean Absolute Error' : cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error').mean() * -1 ,

                                      'Time' : cv_results['fit_time'].mean(),

                                     'Output' : 'y_pred_decision_tree'}, ignore_index=1)
display(cv_results)
# #Graph MLA version of Decision Tree: http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html

# import graphviz 

# dot_data = clf.export_graphviz(dtree, out_file=None, 

#                                 feature_names = data1_x_bin, class_names = True,

#                                 filled = True, rounded = True)

# graph = graphviz.Source(dot_data) 

# graph
clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, y_train)



y_pred_random_forest = clf.predict(X_test)

acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)

print (acc_random_forest)



cv_scores = cross_val_score(clf, X_train, y_train, scoring='accuracy')

print('Model Accuracy %.2f' %(cv_scores.mean()*100))



scores = cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error')

print('Mean Absolute Error %2f' %(-1 * scores.mean()))

mae_rforest = -1 * scores.mean()





# CV Results

cv_results = model_selection.cross_validate(clf, X_train, y_train, cv  = cv_split)

# Appending to Algorithms evaluation DF

Algorithms =  Algorithms.append( {'ML Algorithm' : clf.__class__.__name__ , 

                                     'Parameters': clf.get_params(),

                                     'Train Accuracy Mean' : cv_results['train_score'].mean(), 

                                     'Test Accuracy Mean' : cv_results['test_score'].mean(), 

                                  'Mean Absolute Error' : cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error').mean() * -1 ,

                                     'Test Accuracy 3*STD' : cv_results['test_score'].std()*3,

                                      'Time' : cv_results['fit_time'].mean(),

                                     'Output' : 'y_pred_random_forest'}, ignore_index=1)
# Loading GridSearchCV 

from sklearn.model_selection import GridSearchCV



# create param grid object 

forrest_params = dict(     

    max_depth = [n for n in range(9, 14)],     

    min_samples_split = [n for n in range(4, 11)], 

    min_samples_leaf = [n for n in range(2, 5)],     

    n_estimators = [n for n in range(10, 60, 10)],

#      n_jobs= [n for n in range(3, 3, 1)]

    n_jobs= [3]

)
# # Instiatie Random Forest 

# clf = RandomForestClassifier()



# # Build a GridSearchCV

# forest_cv = GridSearchCV(estimator=clf, param_grid=forrest_params, cv=5) 



# # Fitting to find the best_score & best_estimator_

# forest_cv.fit(X_train, y_train)
# print("Best score: {}".format(forest_cv.best_score_))

# print("Optimal params: {}".format(forest_cv.best_estimator_))



# Best score: 0.8439955106621774

# Optimal params: RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

#             max_depth=12, max_features='auto', max_leaf_nodes=None,

#             min_impurity_decrease=0.0, min_impurity_split=None,

#             min_samples_leaf=3, min_samples_split=6,

#             min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=3,

#             oob_score=False, random_state=None, verbose=0,

#             warm_start=False)
## Using the best parameters to avoid the time penality 

forest_cv = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=12, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=3, min_samples_split=6,

            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=3,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)



# Fitting to find the best_score & best_estimator_

forest_cv.fit(X_train, y_train)
y_pred_grid_rforest = forest_cv.predict(X_test)

acc_grid_rforest = round(forest_cv.score(X_train, y_train) * 100, 2)

print (acc_grid_rforest)



# cv_scores = cross_val_score(forest_cv, X_train, y_train, scoring='accuracy')

# print('Model Accuracy %.2f' %(cv_scores.mean()*100))



scores = cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error')

print('Mean Absolute Error %2f' %(-1 * scores.mean()))

mae_grid_rforest = -1 * scores.mean()





# CV Results

cv_results = model_selection.cross_validate(clf, X_train, y_train, cv  = cv_split)

# Appending to Algorithms evaluation DF

Algorithms =  Algorithms.append( {'ML Algorithm' : 'RandomForest_GridCV' , 

                                     'Parameters': clf.get_params(),

                                     'Train Accuracy Mean' : cv_results['train_score'].mean(), 

                                     'Test Accuracy Mean' : cv_results['test_score'].mean(), 

                                  'Mean Absolute Error' : cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error').mean() * -1 ,

                                     'Test Accuracy 3*STD' : cv_results['test_score'].std()*3,

                                      'Time' : cv_results['fit_time'].mean(),

                                     'Output' : 'y_pred_grid_rforest'}, ignore_index=1)
clf = GaussianNB()

clf.fit(X_train, y_train)

y_pred_gnb = clf.predict(X_test)

acc_gnb = round(clf.score(X_train, y_train) * 100, 2)

print (acc_gnb)



cv_scores = cross_val_score(clf, X_train, y_train, scoring='accuracy')

print('Model Accuracy %.2f' %(cv_scores.mean()*100))



scores = cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error')

print('Mean Absolute Error %2f' %(-1 * scores.mean()))

mae_nbayes = -1 * scores.mean()





# CV Results

cv_results = model_selection.cross_validate(clf, X_train, y_train, cv  = cv_split)

# Appending to Algorithms evaluation DF

Algorithms =  Algorithms.append( {'ML Algorithm' : clf.__class__.__name__ , 

                                     'Parameters': clf.get_params(),

                                     'Train Accuracy Mean' : cv_results['train_score'].mean(), 

                                     'Test Accuracy Mean' : cv_results['test_score'].mean(), 

                                  'Mean Absolute Error' : cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error').mean() * -1 ,

                                     'Test Accuracy 3*STD' : cv_results['test_score'].std()*3,

                                      'Time' : cv_results['fit_time'].mean(),

                                     'Output' : 'y_pred_gnb'}, ignore_index=1)
clf = Perceptron(max_iter=5, tol=None)

clf.fit(X_train, y_train)

y_pred_perceptron = clf.predict(X_test)

acc_perceptron = round(clf.score(X_train, y_train) * 100, 2)

print (acc_perceptron)



cv_scores = cross_val_score(clf, X_train, y_train, scoring='accuracy')

print('Model Accuracy %.2f' %(cv_scores.mean()*100))



scores = cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error')

print('Mean Absolute Error %2f' %(-1 * scores.mean()))

mae_perc = -1 * scores.mean()





# CV Results

cv_results = model_selection.cross_validate(clf, X_train, y_train, cv  = cv_split)

# Appending to Algorithms evaluation DF

Algorithms =  Algorithms.append( {'ML Algorithm' : clf.__class__.__name__ , 

                                     'Parameters': clf.get_params(),

                                     'Train Accuracy Mean' : cv_results['train_score'].mean(), 

                                     'Test Accuracy Mean' : cv_results['test_score'].mean(), 

                                  'Mean Absolute Error' : cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error').mean() * -1 ,

                                     'Test Accuracy 3*STD' : cv_results['test_score'].std()*3,

                                      'Time' : cv_results['fit_time'].mean(),

                                     'Output' : 'y_pred_perceptron'}, ignore_index=1)
clf = SGDClassifier(max_iter=5, tol=None)

clf.fit(X_train, y_train)

y_pred_sgd = clf.predict(X_test)



acc_sgd = round(clf.score(X_train, y_train) * 100, 2)

print (acc_sgd)



cv_scores = cross_val_score(clf, X_train, y_train, scoring='accuracy')

print('Model Accuracy %.2f' %(cv_scores.mean()*100))



scores = cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error')

print('Mean Absolute Error %2f' %(-1 * scores.mean()))

mae_sgd = -1 * scores.mean()



# CV Results

cv_results = model_selection.cross_validate(clf, X_train, y_train, cv  = cv_split)

# Appending to Algorithms evaluation DF

Algorithms =  Algorithms.append( {'ML Algorithm' : clf.__class__.__name__ , 

                                     'Parameters': clf.get_params(),

                                     'Train Accuracy Mean' : cv_results['train_score'].mean(), 

                                     'Test Accuracy Mean' : cv_results['test_score'].mean(), 

                                  'Mean Absolute Error' : cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error').mean() * -1 ,

                                     'Test Accuracy 3*STD' : cv_results['test_score'].std()*3,

                                      'Time' : cv_results['fit_time'].mean(),

                                     'Output' : 'y_pred_sgd'}, ignore_index=1)
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split



# Preping Data

# Here we are using train_X and train_y not to mess with other models

gb_X_train, gb_X_test, gb_y_train, gb_y_test = train_test_split(X_train, y_train, test_size=0.25)



clf = XGBClassifier()

# Add silent=True to avoid printing out updates with each cycle

clf.fit(gb_X_train, gb_y_train, verbose=True)





#We similarly evaluate a model and make predictions as we would do in scikit-learn.

# make predictions

# predictions = clf.predict(test_X)
clf = XGBClassifier(n_estimators=1000)

clf.fit(gb_X_train, gb_y_train, early_stopping_rounds=10, 

             eval_set=[(gb_X_train, gb_y_train)], verbose=False)
my_model = XGBClassifier(n_estimators=1000, learning_rate=0.05)

my_model.fit(gb_X_train, gb_y_train, early_stopping_rounds=10, 

             eval_set=[(gb_X_test, gb_y_test)], verbose=False)
# Model

clf = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=1, gamma=0, importance_type='gain',

       learning_rate=0.05, max_delta_step=0, max_depth=3,

       min_child_weight=1, missing=None, n_estimators=2000, n_jobs=3,

       nthread=-1, objective='binary:logistic', random_state=0, reg_alpha=0,

       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,

       subsample=1)

        

# param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' , 'early_stopping_rounds':10 }

# param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'reg:linear' , 'early_stopping_rounds':10 }



# num_round = 2



# Fitting

clf.fit(X_train, y_train, early_stopping_rounds=10, eval_set= [(X_train, y_train)], verbose = False )



print( clf.get_xgb_params )



# Model Predications

y_pred_xgb = clf.predict(X_test)



# Results

acc_xgb = round(clf.score(X_train, y_train) * 100, 2)

print (acc_xgb)



scores = cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error')

print('Mean Absolute Error %2f' %(-1 * scores.mean()))

mae_xgb = -1 * scores.mean()



# CV Results

cv_results = model_selection.cross_validate(clf, X_train, y_train, cv  = cv_split)

# Appending to Algorithms evaluation DF

Algorithms =  Algorithms.append( {'ML Algorithm' : clf.__class__.__name__ , 

                                     'Parameters': clf.get_params(),

                                     'Train Accuracy Mean' : cv_results['train_score'].mean(), 

                                     'Test Accuracy Mean' : cv_results['test_score'].mean(), 

                                  'Mean Absolute Error' : cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error').mean() * -1 ,

                                     'Test Accuracy 3*STD' : cv_results['test_score'].std()*3,

                                      'Time' : cv_results['fit_time'].mean(),

                                     'Output' : 'y_pred_xgb'}, ignore_index=1)
# features = [ "your list of features ..." ]

# mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))

# print(mapFeat, clf.booster().get_fscore() )

# ts = pd.Series(clf.booster().get_fscore())

# ts.index = ts.reset_index()['index'].map(mapFeat)

# ts.order()[-15:].plot(kind="barh", title=("features importance"))
from sklearn.metrics import confusion_matrix

import itertools



# clf = RandomForestClassifier(n_estimators=100)

# clf.fit(X_train, y_train)

y_pred_random_forest_training_set = forest_cv.predict(X_train)

acc_random_forest = round(forest_cv.score(X_train, y_train) * 100, 2)

print ("Accuracy: %i %% \n"%acc_random_forest)



class_names = ['Survived', 'Not Survived']



# Compute confusion matrix

cnf_matrix = confusion_matrix(y_train, y_pred_random_forest_training_set)

np.set_printoptions(precision=2)



print ('Confusion Matrix in Numbers')

print (cnf_matrix)

print ('')



cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]



print ('Confusion Matrix in Percentage')

print (cnf_matrix_percent)

print ('')



true_class_names = ['True Survived', 'True Not Survived']

predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']



df_cnf_matrix = pd.DataFrame(cnf_matrix, 

                             index = true_class_names,

                             columns = predicted_class_names)



df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, 

                                     index = true_class_names,

                                     columns = predicted_class_names)



plt.figure(figsize = (15,5))



plt.subplot(121)

sns.heatmap(df_cnf_matrix, annot=True, fmt='d')



plt.subplot(122)

sns.heatmap(df_cnf_matrix_percent, annot=True)
# models = pd.DataFrame({

#     'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 

#               'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 

#               'Perceptron', 'Stochastic Gradient Decent', 'XGBoost Classifier', 'Grid Random Forest'],

    

#     'Score': [acc_log_reg, acc_svc, acc_linear_svc, 

#               acc_knn,  acc_decision_tree, acc_random_forest, acc_gnb, 

#               acc_perceptron, acc_sgd, acc_xgb, acc_grid_rforest],

    

#     'MAE' : [mae_log_reg, mae_svc, mae_lsvc, 

#              mae_knn, mae_decision_tree, mae_rforest, mae_nbayes,

#              mae_perc, mae_sgd, mae_xgb, mae_grid_rforest],

    

#     'Output': ['y_pred_log_reg' , 'y_pred_svc' , 'y_pred_linear_svc' ,

#                'y_pred_knn' , 'y_pred_decision_tree' , 'y_pred_random_forest' ,'y_pred_gnb' ,

#                'y_pred_perceptron' , 'y_pred_sgd' , 'y_pred_xgb' , 'y_pred_grid_rforest']

#     })



# models.sort_values(by='Score', ascending=False)

with pd.option_context('display.max_rows', 20, 'display.max_columns', 25, 'display.max_colwidth', 360):    

    display(Algorithms.sort_values(by =['Test Accuracy Mean'], ascending = 0) )

    display(Algorithms.sort_values(by =['Test Accuracy 3*STD'], ascending = 0) )
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred_grid_rforest

    })



submission.to_csv('submission.csv', index=False)



# Random Forest = 0.77

# XGBoost  = 0.79904

# Grid Random Forest = 0.79904