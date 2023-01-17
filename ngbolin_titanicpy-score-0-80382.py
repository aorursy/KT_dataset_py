%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
combined = pd.concat([df_train, df_test])
combined.head()
combined.apply(set) # Unique values that the features can take on
def first_name(x): 

    return x.split(', ')[0]



combined['FirstName'] = combined['Name'].apply(first_name)
def title(x):

    return x.split(', ')[1].split('.')[0]



combined['Title'] = combined['Name'].apply(title)
combined.head()
print(combined['Title'].value_counts())

print(len(set(combined['Title'])))
def unique_title(x):

    if x in ('Mr', 'Miss', 'Mrs', 'Master'): return x

    elif x in ('Ms', 'Mlle', 'Mme'): return 'Miss'

    else: return 'Others'

    

combined['Title'] = combined['Title'].apply(unique_title)
plt.figure(figsize=(15, 10))

plt.subplot(1, 1, 1)

plt.hist(combined['Age'].dropna(), bins=range(0, 81, 1), color = 'green')

plt.title('Age Distribution of Combined Dataset')

plt.show()
plt.figure(figsize=(15, 10))

plt.subplot(1, 1, 1)

sns.distplot(combined[combined.Survived == 0].Age.dropna(), bins=range(0, 81, 1), 

             color = 'red', label = 'Perished')

sns.distplot(combined[combined.Survived == 1].Age.dropna(), 

             bins=range(0, 81, 1), color = 'blue', label = 'Survived')



plt.title('Age Distribution of Survivors')

plt.legend()

plt.show()
combined['Child'] = (combined['Age'] <= 10)
print('Total Unique Values for Ticket: ', len(set(combined['Ticket'])))

print('Length of Combined Dataset: ', len(combined))
shared_tickets = combined.groupby('Ticket')['Name'].transform('count')



plt.figure(figsize=(15, 10))

plt.hist(shared_tickets.values)

plt.show()



combined['SharedTickets'] = shared_tickets.values
plt.figure(figsize=(20, 10))



sns.distplot(combined[(combined.Survived == 0)].SharedTickets.dropna(),

             bins = range(1, 12, 1), kde = False, norm_hist = True, 

             color = 'red', label = 'Perished')

sns.distplot(combined[(combined.Survived == 1)].SharedTickets.dropna(),

             bins = range(1, 12, 1), kde = False, norm_hist = True, 

             color = 'blue', label = 'Survived')



plt.title('Distribution of Shared Tickets')

plt.legend()

plt.show()
def shared_tickets(x): 

    if x > 1: return(0)

    else: return(1)



def good_shared_tickets(x):

    if x > 1 and x < 5: return(1)

    else: return(0)



combined['GoodSharedTickets'] = combined['SharedTickets'].apply(good_shared_tickets)

combined['Alone'] = combined['SharedTickets'].apply(shared_tickets)
combined['TicketType'] = combined['Ticket'].apply(lambda x: x[0])



print('Unique Ticket Values: ', len(set(combined['Ticket'])))

print('Unique Ticket Values (First Character): ', len(set(combined['TicketType'])))
combined.isnull().sum()
print(len(set(combined['Cabin'])))
print(combined['Cabin'].value_counts()[0:5])
def cabintostay(x):

    if pd.isnull(x): return(0)

    else: return(1)



combined['CabinToStay'] = combined['Cabin'].apply(cabintostay)
def cabinclass(x):

    if pd.isnull(x): return('N')

    else: return (x.split())[0][0]



combined['CabinClass'] = combined['Cabin'].apply(cabinclass)
combined.describe()
plt.figure(figsize=(20,10))



plt.subplot(121)

sns.distplot(combined[combined.Survived == 0].Parch.dropna(), bins=range(0, 10, 1), 

             kde = False, norm_hist = True, 

             color = 'red', label = 'Perished')

sns.distplot(combined[combined.Survived == 1].Parch.dropna(), bins=range(0, 10, 1), 

             kde = False, norm_hist = True, 

             color = 'blue', label = 'Survived')

plt.ylim([0, 1])



plt.subplot(122)

sns.distplot(combined[combined.Survived == 0].SibSp.dropna(), bins=range(0, 10, 1), 

             kde = False, norm_hist = True, 

             color = 'red', label = 'Perished')

sns.distplot(combined[combined.Survived == 1].SibSp.dropna(), bins=range(0, 10, 1), 

             kde = False, norm_hist = True, 

             color = 'blue', label = 'Survived')



plt.ylim([0, 1])

plt.legend()

plt.show()
combined['Family'] = combined['Parch'] + combined['SibSp']
plt.figure(figsize=(20,10))



sns.distplot(combined[combined.Survived == 0].Family.dropna(), bins=range(0, 12, 1), 

             kde = False, norm_hist = True, 

             color = 'red', label = 'Perished')

sns.distplot(combined[combined.Survived == 1].Family.dropna(), bins=range(0, 12, 1), 

             kde = False, norm_hist = True, 

             color = 'blue', label = 'Survived')



plt.legend()

plt.show()
combined['MiddleFam'] = ((combined['Family'] > 0) & (combined['Family'] < 4))
plt.figure(figsize=(20,8))



plt.subplot(1, 2, 1)

sns.distplot(combined[combined.Survived == 0].Fare.dropna(), bins = range(0, 500, 10),

             norm_hist = True, color = 'red', label = 'Perished')

sns.distplot(combined[combined.Survived == 1].Fare.dropna(), bins = range(0, 500, 10),

             norm_hist = True, color = 'blue', label = 'Survived')

plt.legend()



plt.subplot(1, 2, 2)

sns.barplot('Embarked', 'Survived', data = combined)



plt.show()
fig = plt.figure(figsize=(20,8))

ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)



sns.boxplot('Embarked', 'Fare', data = combined, ax = ax1)



tab1 = pd.crosstab(combined['Embarked'], combined['Sex'])

plot1 = tab1.div(tab1.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax2)



plt.show()
plt.figure(figsize=(20,10))



plt.subplot(121)

sns.distplot(combined[(combined.Survived == 0) & (combined.Sex == 'male')].Fare.dropna(), 

             bins=range(0, 500, 10),  kde = False, norm_hist = True, 

             color = 'red', label = 'Perished')

sns.distplot(combined[(combined.Survived == 1) & (combined.Sex == 'male')].Fare.dropna(), 

             bins=range(0, 500, 10),  kde = False, norm_hist = True, 

             color = 'blue', label = 'Survived')



plt.legend()

plt.ylim([0, 0.06])

plt.title('Fare Distribution of Survivors and Non-survivors [Males]')



plt.subplot(122)

sns.distplot(combined[(combined.Survived == 0) & (combined.Sex == 'female')].Fare.dropna(), 

             bins=range(0, 500, 10),  kde = False, norm_hist = True, 

             color = 'red', label = 'Perished')

sns.distplot(combined[(combined.Survived == 1) & (combined.Sex == 'female')].Fare.dropna(), 

             bins=range(0, 500, 10),  kde = False, norm_hist = True, 

             color = 'blue', label = 'Survived')



plt.legend()

plt.ylim([0, 0.06])

plt.title('Fare Distribution of Survivors and Non-survivors [Females]')



plt.show()



print ((combined[(combined.Survived == 0) & (combined.Sex == 'male')].Fare.mean()), 

       (combined[(combined.Survived == 1) & (combined.Sex == 'male')].Fare.mean()))

print ((combined[(combined.Survived == 0) & (combined.Sex == 'female')].Fare.mean()), 

       (combined[(combined.Survived == 1) & (combined.Sex == 'female')].Fare.mean()))
print('Number of Tickets costing more than $500: ', sum(combined.Fare.dropna() > 500))
combined[combined.Fare > 500] # What are the 4 rows?
plt.figure(figsize=(20,8))



plt.subplot(121)

sns.distplot(combined[(combined.Alone == 0) & (combined.Survived == 0)].Fare.dropna(),

             bins=range(0, 500, 10),  kde = False, norm_hist = True, 

             color = 'red', label = 'Perished')

sns.distplot(combined[(combined.Alone == 0) & (combined.Survived == 1)].Fare.dropna(),

             bins=range(0, 500, 10),  kde = False, norm_hist = True, 

             color = 'blue', label = 'Survived')



plt.legend()

plt.title('Shared Tickets Fare Distribution')



plt.subplot(122)

sns.distplot(combined[(combined.Alone == 1) & (combined.Survived == 0)].Fare.dropna(),

             bins=range(0, 500, 10),  kde = False, norm_hist = True, 

             color = 'red', label = 'Perished')

sns.distplot(combined[(combined.Alone == 1) & (combined.Survived == 1)].Fare.dropna(),

             bins=range(0, 500, 10),  kde = False, norm_hist = True, 

             color = 'blue', label = 'Survived')



plt.legend()

plt.title('Normal Tickets Fare Distribution (Passengers who travelled alone)')



plt.show()
from scipy import stats



stats.ks_2samp(combined[(combined.Alone == 0) & (combined.Survived == 0)].Fare.dropna(),

               combined[(combined.Alone == 1) & (combined.Survived == 0)].Fare.dropna())
shared_tickets = combined.groupby('Ticket')['Name'].transform('count')

combined['AdjustedFare'] = combined['Fare'] / shared_tickets
plt.figure(figsize=(20,8))



plt.subplot(121)

sns.distplot(combined[(combined.Alone == 0) & (combined.Survived == 0)].AdjustedFare.dropna(),

             bins=range(0, 500, 10),  kde = False, norm_hist = True, 

             color = 'red', label = 'Perished')

sns.distplot(combined[(combined.Alone == 0) & (combined.Survived == 1)].AdjustedFare.dropna(),

             bins=range(0, 500, 10),  kde = False, norm_hist = True, 

             color = 'blue', label = 'Survived')



plt.legend()

plt.title('Shared Tickets Adjusted Fare Distribution')



plt.subplot(122)

sns.distplot(combined[(combined.Alone == 1) & (combined.Survived == 0)].AdjustedFare.dropna(),

             bins=range(0, 500, 10),  kde = False, norm_hist = True, 

             color = 'red', label = 'Perished')

sns.distplot(combined[(combined.Alone == 1) & (combined.Survived == 1)].AdjustedFare.dropna(),

             bins=range(0, 500, 10),  kde = False, norm_hist = True, 

             color = 'blue', label = 'Survived')



plt.legend()

plt.title('Normal Tickets Adjusted Fare Distribution (Passengers who travelled alone)')



plt.show()
plt.figure(figsize=(20,8))

sns.distplot(combined[(combined.Survived == 0)].AdjustedFare.dropna(),

             bins=range(0, 150, 1), kde = False, norm_hist = True, 

             color = 'red', label = 'Perished')

sns.distplot(combined[(combined.Survived == 1)].AdjustedFare.dropna(),

             bins=range(0, 150, 1), kde = False, norm_hist = True, 

             color = 'blue', label = 'Survived')



plt.legend()

plt.title('Adjusted Fare Distribution')

plt.show()
combined['CheapTickets'] = combined['AdjustedFare'] <= 10
plt.figure(figsize=(20,8))

sns.pointplot('Sex', 'Survived', hue = 'Pclass', data = combined)

plt.show()
fig = plt.figure(figsize=(20,8))

ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)



p = sns.factorplot('Embarked', 'Survived', hue = 'Sex', data = combined, ax = ax1)

plt.close(p.fig)



g = sns.factorplot('Embarked', 'Survived', hue = 'Pclass', data = combined, ax = ax2)

plt.close(g.fig)
fig = plt.figure(figsize=(20,8))

ax1 = fig.add_subplot(221); ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(223); ax4 = fig.add_subplot(224)



tab1 = pd.crosstab(combined['Embarked'], combined['Sex'])

plot1 = tab1.div(tab1.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax1)

plot1 = plt.xlabel('Embarkation Point')

plot1 = plt.ylabel('Percentage')

plot1 = plt.title('Sex')



tab2 = pd.crosstab(combined['Embarked'], combined['Pclass'])

plot2 = tab2.div(tab2.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax2)

plot2 = plt.xlabel('Embarkation Point')

plot2 = plt.ylabel('Percentage')

plot2 = plt.title('Pclass')



tab3 = pd.crosstab(combined['Embarked'], combined['GoodSharedTickets'])

plot3 = tab3.div(tab3.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax3)

plot3 = plt.xlabel('Embarkation Point')

plot3 = plt.ylabel('Percentage')

plot3 = plt.title('Good Shared Tickets')



tab4 = pd.crosstab(combined['Embarked'], combined['CabinToStay'])

plot4 = tab4.div(tab3.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax4)

plot4 = plt.xlabel('Embarkation Point')

plot4 = plt.ylabel('Percentage')

plot4 = plt.title('Cabin To Stay')
fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(221); ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(223); ax4 = fig.add_subplot(224)



g1 = sns.factorplot('CabinToStay', 'Survived', hue = 'Pclass', data = combined, ax = ax1)

plt.close(g1.fig)

sns.violinplot('CabinToStay', 'CheapTickets', hue = 'Survived', 

               data = combined, ax = ax2, split = True)



tab1 = pd.crosstab(combined['CabinToStay'].dropna(), 

                   combined[pd.notnull(combined.CabinToStay)].Sex)

tab1.div(tab1.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax3)



tab2 = pd.crosstab(combined['CabinToStay'].dropna(), 

                   combined[pd.notnull(combined.CabinToStay)].Pclass)

tab2.div(tab2.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax4)



plt.show()
fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(211); ax2 = fig.add_subplot(212)



g1 = sns.factorplot('CabinClass', 'Survived', 

                    kind = 'bar', data = combined, ax = ax1)

plt.close(g1.fig)

g2 = sns.factorplot('TicketType', 'Survived', 

                    kind = 'bar', data = combined, ax = ax2)

plt.close(g2.fig)

plt.show()
fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(221); ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(223); ax4 = fig.add_subplot(224)



tab1 = pd.crosstab(combined['CabinClass'], 

                   combined['Sex'])

dummy1 = tab1.div(tab1.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax1)

dummy1 = plt.xlabel('Cabin Class')

dummy1 = plt.ylabel('Percentage')



tab2 = pd.crosstab(combined['CabinClass'], 

                   combined['Pclass'])

dummy2 = tab2.div(tab2.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax2)

dummy2 = plt.xlabel('Cabin Class')

dummy2 = plt.ylabel('Percentage')



tab3 = pd.crosstab(combined['TicketType'], 

                   combined['Sex'])

dummy3 = tab3.div(tab3.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax3)

dummy3 = plt.xlabel('Ticket Type')

dummy3 = plt.ylabel('Percentage')



tab4 = pd.crosstab(combined['TicketType'], 

                   combined['Pclass'])

dummy4 = tab4.div(tab4.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax4)

dummy4 = plt.xlabel('Ticket Type')

dummy4 = plt.ylabel('Percentage')



plt.show()
tickettype_groupby = combined.groupby('TicketType')['Survived'].apply(np.mean)



def good_ticket(ticket_type):

    if tickettype_groupby[ticket_type] > 0.4: return(1)

    else: return(0)

    

combined['GoodTicket'] = combined['TicketType'].apply(good_ticket)
cabinclass_groupby = combined.groupby('CabinClass')['Survived'].apply(np.mean)



def good_cabin(cabinclass):

    if cabinclass_groupby[cabinclass] > 0.55: return(1)

    else: return(0)

    

combined['GoodCabinClass'] = combined['CabinClass'].apply(good_cabin)
train = combined[:len(df_train)]

test = combined[len(df_train):]
combined.isnull().sum()
plt.figure(figsize=(30, 12))



plt.subplot(211)

train_male = train[train.Sex == 'male']

sns.heatmap(train_male.corr(), annot = True)

plt.title('Correlation Plot for Males')



plt.subplot(212)

train_female = train[train.Sex == 'female']

sns.heatmap(train_female.corr(), annot = True)

plt.title('Correlation Plot for Females')



plt.show()
age_prior = combined['Age'].copy()

groupby_impute_age = train.groupby(['Title']).apply(np.mean)[['Age']]

groupby_impute_age
def return_age(age, title):

    if pd.notnull(age): return age

    else: return groupby_impute_age.ix[title]



return_age_vec = np.vectorize(return_age)

train['Age'] = return_age_vec(train['Age'], train['Title'])

test['Age'] = return_age_vec(test['Age'], test['Title'])
plt.figure(figsize=(15, 10))

plt.subplot(1, 1, 1)



sns.distplot(combined['Age'], bins=range(0, 81, 1), color = 'orange')

sns.distplot(age_prior.dropna(), bins=range(0, 81, 1), color = 'green')

plt.title('Age Distribution of Combined Dataset')



plt.show()
train[train.Embarked.isnull()]
train.ix[61, 'Embarked'] = 'C'

train.ix[829, 'Embarked'] = 'C'
test[test.Fare.isnull()]
print(combined.groupby(['Pclass', 'CabinToStay', 

                        'Embarked'])['Fare'].apply(np.mean).ix[3].ix[0].ix['S'])
test.ix[152, 'Fare'] = combined.groupby(['Pclass', 'CabinToStay', 

                                         'Embarked'])['Fare'].apply(np.mean).loc[3].loc[0].loc['S']
# Adjusted Fare is the same as Fare since passenger is alone

test.ix[152, 'AdjustedFare'] = test.ix[152, 'Fare']

test.ix[152, 'CheapTickets'] = test.ix[152, 'AdjustedFare'] <= 0
plt.figure(figsize=(20, 12))

sns.heatmap(combined[:len(df_train)].corr(), annot = True)

plt.show()
from sklearn.preprocessing import LabelEncoder

encodeFeatures = ['Title', 'Pclass', 'Sex', 'Child', 'SharedTickets', 'Alone', 'Embarked',

                  'Family',  'GoodCabinClass', 'GoodTicket', 'CheapTickets', 'MiddleFam']



for i in encodeFeatures:

    combined[i] = combined[i].astype('category')

    

le = LabelEncoder()

combined_processed = combined[encodeFeatures].apply(le.fit_transform)
# Splitting Training Data

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



X = combined_processed

y = combined['Survived']



X_test = X[len(train):].copy()

X_train = X[:len(train)].copy(); y_train = y[:len(train)].copy()



X_subtrain, X_subtest, y_subtrain, y_subtest = train_test_split(X_train, y_train, 

                                                                test_size = 0.2,

                                                                random_state = 42)
# Set Random State

random_state = 1212
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier



rf = RandomForestClassifier(n_estimators = 500, random_state = random_state).fit(X_subtrain, y_subtrain)

gb = GradientBoostingClassifier(n_estimators = 300, random_state = random_state).fit(X_subtrain, y_subtrain)
feature_importance = pd.DataFrame()

feature_importance['Features'] = X_subtrain.columns

feature_importance['RandomForest'] = rf.fit(X_subtrain, y_subtrain).feature_importances_

feature_importance['GBM'] = gb.fit(X_subtrain, y_subtrain).feature_importances_

feature_importance
# Model 1: Support Vector Machine

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV



svm = SVC(random_state = random_state, probability = True)

param_grid = {'kernel': ['linear', 'rbf'],

              'C': np.logspace(0, 2, 20)}

svm_clf = GridSearchCV(svm, param_grid).fit(X_subtrain, y_subtrain)



svm_score = cross_val_score(svm_clf, X_subtrain, y_subtrain, cv = 5).mean()

print(svm_score)
# Model 2: Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 1000,

                            min_samples_split=10,

                            random_state = random_state)



param_grid = {'criterion': ['gini', 'entropy'],

              'max_depth': [4, 5, 6]}

rf_clf = GridSearchCV(rf, param_grid).fit(X_subtrain, y_subtrain)



rf_score = cross_val_score(rf_clf, X_subtrain, y_subtrain, cv = 5).mean()

print(rf_score)
# Model 3: Gradient Boosting Model

from sklearn.ensemble import GradientBoostingClassifier



gb = GradientBoostingClassifier(n_estimators = 500, 

                                max_depth = 1, random_state = random_state)



param_grid = {'learning_rate': np.logspace(-2, 2, 10),

              'loss': ['deviance', 'exponential']}

gb_clf = GridSearchCV(gb, param_grid).fit(X_subtrain, y_subtrain)



gb_score = cross_val_score(gb_clf, X_subtrain, y_subtrain, cv = 5).mean()

print(gb_score)
# Model 4: K-Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

param_grid = {'n_neighbors': [3, 5, 7, 10],

              'weights': ['uniform', 'distance']}

knn_clf = GridSearchCV(knn, param_grid).fit(X_subtrain, y_subtrain)



knn_score = cross_val_score(knn_clf, X_subtrain, y_subtrain, cv = 5).mean()

print(knn_score)
print('Training Score - SVM: ', svm_score)

print('Testing Score - SVM: ',svm_clf.score(X_subtest, y_subtest))



print('\n')



print('Training Score - Random Forest Classifier: ', rf_score)

print('Testing Score - Random Forest: ', rf_clf.score(X_subtest, y_subtest))



print('\n')



print('Training Score - Gradient Boosting Classifier: ', gb_score)

print('Testing Score - Gradient Boosting Classifier: ', gb_clf.score(X_subtest, y_subtest))



print('\n')



print('Training Score - K-Nearest Neighbors Classifier: ', knn_score)

print('Testing Score - K-Nearest Neighbors Classifier: ', knn_clf.score(X_subtest, y_subtest))
cv = pd.DataFrame()

cv['SVM'] = svm_clf.predict(X_subtrain)

cv['Random Forest'] = rf_clf.predict(X_subtrain)

cv['Gradient Boosting'] = gb_clf.predict(X_subtrain)

cv['K-Nearest Neighbors'] = knn_clf.predict(X_subtrain)
plt.figure(figsize=(20, 15))

sns.heatmap(cv.corr(), annot = True)

plt.show()
cv_score = pd.DataFrame(index=['Max Cross-Validation Score', 'Testing Score'])

cv_score['SVM'] = [svm_score, svm_clf.score(X_subtest, y_subtest)]

cv_score['Random Forest'] = [rf_score, rf_clf.score(X_subtest, y_subtest)]

cv_score['Gradient Boosting'] = [gb_score, gb_clf.score(X_subtest, y_subtest)]
no_of_models = range(1, len(cv_score.columns) + 1)



plt.figure(figsize = (20, 8))



plt.plot(no_of_models, cv_score.T['Max Cross-Validation Score'], 

         no_of_models, cv_score.T['Testing Score'])



plt.legend(['Mean Cross-Validation Score', 'Testing Score'], loc='lower right')



plt.xticks(no_of_models, cv_score.T.index)

plt.ylim([0, 1])

plt.show()
cv_pred = pd.DataFrame()

cv_pred['SVM'] = svm_clf.predict(X_subtrain)

cv_pred['Random Forest'] = rf_clf.predict(X_subtrain)

cv_pred['Gradient Boosting'] = gb_clf.predict(X_subtrain)
from sklearn.metrics import accuracy_score



def pred(x):

    if x >= 2: return 1

    else: return 0

    

print(accuracy_score(y_subtrain, list(map(pred, cv_pred.sum(axis = 1)))))
cv_pred = pd.DataFrame()

cv_pred['SVM'] = svm_clf.predict(X_subtest)

cv_pred['Random Forest'] = rf_clf.predict(X_subtest)

cv_pred['Gradient Boosting'] = gb_clf.predict(X_subtest)
from sklearn.metrics import accuracy_score

ensemble = map(pred, cv_pred.sum(axis = 1))

print(accuracy_score(y_subtest, cv_pred['Random Forest']))
pred = rf_clf.predict(X_test)
print('Mean Survival Rate for Training Dataset: ', np.mean(df_train['Survived']))

print('Mean Survival Rate for Testing Dataset: ', np.mean(pred))
submission = pd.read_csv('../input/genderclassmodel.csv')

submission['Survived'] = list(map(int, pred))

submission.head()
submission.to_csv('submission.csv', index = False)