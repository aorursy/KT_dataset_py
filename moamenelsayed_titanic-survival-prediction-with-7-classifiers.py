# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



from sklearn.model_selection import GridSearchCV,RandomizedSearchCV





train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.info()
test.info()
frames = [train, test]

all_data = pd.concat(frames, axis=0)

all_data.shape
all_data.tail()
plt.figure(figsize=(15,8))



ax = sns.countplot(x='Pclass', data=all_data, palette="Blues_d")

for p in ax.patches:

    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title('# of passengers in each class')

plt.xticks( np.arange(3), ['1st class', '2nd class', '3rd class'])

plt.show()
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

ax = sns.countplot(x='Pclass', hue='Sex', data=all_data, palette="Reds_d")

for p in ax.patches:

    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title('Proportion of male/female in each class')

plt.xticks( np.arange(3), ['1st class', '2nd class', '3rd class'])



plt.subplot(1,2,2)

ax = sns.countplot(x='Pclass', hue='Survived', data=all_data, palette="Blues_d")

for p in ax.patches:

    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title('Proportion of survived/not survived in each class')

plt.xticks( np.arange(3), ['1st class', '2nd class', '3rd class'])

plt.legend(['Not Survived','Survived'], title='Survived')



plt.show()
plt.figure(figsize=(20,8))

ax = sns.catplot(x = 'Pclass', hue = 'Survived', col = 'Sex', kind = 'count', data = all_data, palette='Blues_d')

plt.xticks( np.arange(3), ['1st class', '2nd class', '3rd class'])

plt.show()
plt.figure(figsize=(15,8))



ax = sns.countplot(x='Sex', data=all_data, palette="Blues_d")

for p in ax.patches:

    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title('# of Males and Females')

plt.show()
plt.figure(figsize=(18,5))

binsize=2

bins = np.arange(0,all_data['Age'].max()+binsize,binsize)

sns.distplot(all_data['Age'], bins=bins);

plt.title('Distrubution of passengers age')

plt.xlabel('Age')

plt.ylabel('Frequency');
all_data['Age'].describe()
plt.figure(figsize=(18,5))

palette = 'Blues_d'



plt.subplot(1,3,1)

sns.boxplot(x='Sex', y='Age', data=all_data, palette=palette)



plt.subplot(1,3,2)

sns.boxplot(x='Survived', y='Age', data=all_data, palette=palette)

plt.xticks( np.arange(2), ['not survived', 'survived']);



plt.subplot(1,3,3)

sns.boxplot(x='Pclass', y='Age', data=all_data, palette=palette)

plt.xticks( np.arange(3), ['1st class', '2nd class', '3rd class']);
age_1_class_stat = pd.DataFrame(all_data.groupby('Sex')['Age'].describe())

age_2_class_stat = pd.DataFrame(all_data.groupby('Sex')['Age'].describe())

age_3_class_stat = pd.DataFrame(all_data.groupby('Sex')['Age'].describe())



pd.concat([age_1_class_stat, age_2_class_stat, age_3_class_stat], axis=0, sort = False, keys = ['1st', '2nd', '3rd'])
all_data.isna().sum()
all_data.drop('Cabin', axis=1, inplace=True)
all_data['Age'] = all_data['Age'].fillna(all_data['Age'].mean())
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].mean())

all_data['Embarked'] = all_data['Embarked'].fillna('S')
all_data['Rich_Woman'] = 0

all_data['Rich_Man'] = 0

all_data['Medium_Woman'] = 0

all_data['Medium_Man'] = 0

all_data['Poor_Woman'] = 0

all_data['Poor_Man'] = 0

all_data.loc[(all_data['Pclass']==1) & (all_data['Sex']=='female'), 'Rich_Woman'] = 1

all_data.loc[(all_data['Pclass']==1) & (all_data['Sex']=='male'), 'Rich_Man'] = 1

all_data.loc[(all_data['Pclass']==2) & (all_data['Sex']=='female'), 'Medium_Woman'] = 1

all_data.loc[(all_data['Pclass']==2) & (all_data['Sex']=='female'), 'Medium_Man'] = 1

all_data.loc[(all_data['Pclass']==3) & (all_data['Sex']=='female'), 'Poor_Woman'] = 1

all_data.loc[(all_data['Pclass']==3) & (all_data['Sex']=='male'), 'Poor_Man'] = 1



all_data = pd.concat([all_data, 

                           pd.get_dummies(all_data['Embarked'], 

                           prefix = 'embarked')], axis=1)
all_data['Sex'] = all_data['Sex'].apply(lambda x: 1 if x=='male' else 0)

all_data.head()
all_data['Log_Age'] =  np.log1p(all_data['Age'])

all_data['Log_Fare'] = np.log1p(all_data['Fare'])

all_data.drop(['Age','Fare'], axis=1, inplace=True)
title_dict = {  'Mr':     'Mr',

                'Mrs':    'Mrs',

                'Miss':   'Miss',

                'Master': 'Master',              

                'Ms':     'Miss',

                'Mme':    'Mrs',

                'Mlle':   'Miss',

                'Capt':   'military',

                'Col':    'military',

                'Major':  'military',

                'Dr':     'Dr',

                'Rev':    'Rev',                  

                'Sir':    'honor',

                'the Countess': 'honor',

                'Lady':   'honor',

                'Jonkheer': 'honor',

                'Don':    'honor',

                'Dona':   'honor' }





all_data['Title'] = all_data['Name'].str.split(',', expand = True)[1].str.split('.', expand = True)[0].str.strip(' ')



all_data['Title_category'] = all_data['Title'].map(title_dict)

all_data = pd.concat([all_data, 

                           pd.get_dummies(all_data['Title_category'],

                           prefix = 'title')], axis=1)

print(all_data[(all_data['SibSp']==0)]['Survived'].mean())

print(all_data[(all_data['SibSp']==1)]['Survived'].mean())

print(all_data[(all_data['SibSp']==2)]['Survived'].mean())

print(all_data[(all_data['SibSp']==3)]['Survived'].mean())

print(all_data[(all_data['SibSp']==4)]['Survived'].mean())

print(all_data[(all_data['SibSp']==5)]['Survived'].mean())

print(all_data[(all_data['SibSp']==8)]['Survived'].mean())
all_data['Family_size'] = all_data['SibSp'] + all_data['Parch'] + 1

all_data['Family_size_group'] = all_data['Family_size'].map(lambda x: 'f_single' if x == 1 

                                                                        else ('f_usual' if 5 > x >= 2 

                                                                              else ('f_big' if 8 > x >= 5 

                                                                                   else 'f_large' )))

all_data = pd.concat([all_data, 

                      pd.get_dummies(all_data['Family_size_group'], 

                      prefix = 'family')], axis=1)
del all_data['PassengerId']

del all_data['Ticket']  

del all_data['Title_category']

del all_data['Name']

del all_data['Family_size']

del all_data['Family_size_group'] 

del all_data['Embarked']    
del all_data['Title']
all_data.info()
test_id = test['PassengerId']

train_1 = all_data[(all_data['Survived']==1) | (all_data['Survived']==0)]

test_1 = all_data[(all_data['Survived']!=1) & (all_data['Survived']!=0)]



print(train_1.shape)

print(test_1.shape)
X = train_1.drop('Survived', axis=1)

y = train_1['Survived']
corr=X.corr(method='pearson')

plt.figure(figsize=(25, 25))

mask = np.triu(np.ones_like(corr, dtype=np.bool))

palette = sns.diverging_palette(50, 200, n=256)

sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdBu', mask= mask, vmin=-1, vmax=1, center= 0,

            square=True, linewidths=2, cbar_kws={"shrink": .5})
# Train and validaton errors initialized as empty list

train_errs = list()

valid_errs = list()

C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]



# Loop over values of C_value

for C_value in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:

    # Create LogisticRegression object and fit

    lr = LogisticRegression(C=C_value)

    lr.fit(X, y)

    

    # Evaluate error rates and append to lists

    train_errs.append( 1.0 - lr.score(X, y) )

    valid_errs.append( 1.0 - lr.score(X, y) )

    

# Plot results

plt.semilogx(C_values, train_errs, C_values, valid_errs)

plt.legend(("train", "validation"))

plt.xlabel('C inverse regularization strength')

plt.ylabel('calssifaication errors')

plt.show()
LR = LogisticRegression()

parameters = {'C':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 

              'penalty':['l1','l2']}

searcher_LR = GridSearchCV(LR, parameters, cv=10)

searcher_LR.fit(X, y)



# Report the best parameters and the corresponding score

print("Best CV params", searcher_LR.best_params_)

print("Best CV accuracy", searcher_LR.best_score_)
# Instantiate an RBF SVM

svm = SVC()



# Instantiate the GridSearchCV object and run the search

parameters = {'C':[0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}

searcher_svm = GridSearchCV(svm, parameters)

searcher_svm.fit(X, y)



# Report the best parameters and the corresponding score

print("Best CV params", searcher_svm.best_params_)

print("Best CV accuracy", searcher_svm.best_score_)
# Instantiate an KNN

knn = KNeighborsClassifier()



# Instantiate the GridSearchCV object and run the search

parameters = {'n_neighbors':[1, 3, 5, 7, 9]}

searcher_knn = GridSearchCV(knn, parameters)

searcher_knn.fit(X, y)



# Report the best parameters and the corresponding score

print("Best CV params", searcher_knn.best_params_)

print("Best CV accuracy", searcher_knn.best_score_)
# We set random_state=0 for reproducibility 

linear_classifier = SGDClassifier(random_state=0)



# Instantiate the GridSearchCV object and run the search

parameters = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 

             'loss':['hinge','log'], 'penalty':['l1','l2']}

searcher_SGD = GridSearchCV(linear_classifier, parameters, cv=10)

searcher_SGD.fit(X, y)



# Report the best parameters and the corresponding score

print("Best CV params", searcher_SGD.best_params_)

print("Best CV accuracy", searcher_SGD.best_score_)
# Instantiate an Decision Tree

DT = DecisionTreeClassifier()



# Instantiate the GridSearchCV object and run the search

parameters = {'criterion':['gini','entropy'],    

              'max_depth':[2, 4, 6, 8, 10],

              'min_samples_leaf':[1, 0.1, 0.01, 0.001]

             }

searcher_dt = GridSearchCV(DT, parameters, cv=10)

searcher_dt.fit(X, y)



# Report the best parameters and the corresponding score

print("Best CV params", searcher_dt.best_params_)

print("Best CV accuracy", searcher_dt.best_score_)

#print("Test accuracy of best grid search hypers:", searcher_dt.score(X_test, y_test))
# Instantiate an AdaBoostClassifier

adb_clf = AdaBoostClassifier(base_estimator=DT)



# Instantiate the GridSearchCV object and run the search

parameters = {'n_estimators':[100, 200, 300, 400, 500, 600, 700, 800],

              'learning_rate':[1, 0.1, 0.01, 0.001, 0.0001]

             }

searcher_ada = GridSearchCV(adb_clf, parameters, cv=10)

searcher_ada.fit(X, y)



# Report the best parameters and the corresponding score

print("Best CV params", searcher_ada.best_params_)

print("Best CV accuracy", searcher_ada.best_score_)
# Instantiate an Random Forest Classifier

RF = RandomForestClassifier()



# Instantiate the GridSearchCV object and run the search

parameters = {'n_estimators':[100, 200, 300, 400, 500, 600, 700],

              'min_samples_leaf':[1, 0.1, 0.01, 0.001, 0.0001]

             }

searcher_rf = GridSearchCV(RF, parameters, cv=10)

searcher_rf.fit(X, y)



# Report the best parameters and the corresponding score

print("Best CV params", searcher_rf.best_params_)

print("Best CV accuracy", searcher_rf.best_score_)
test_1.drop('Survived',axis=1, inplace=True)

y_pred = searcher_dt.predict(test_1)

y_pred = y_pred.astype('int64')



submission = pd.DataFrame({'PassengerId': test_id, 'Survived': y_pred})

submission.to_csv("Titanic_submission.csv", index = False)
submission.head()