import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import random, re

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.ensemble import AdaBoostClassifier
test = pd.read_csv('../input/test.csv')

PassengerId = test['PassengerId']

train = pd. read_csv('../input/train.csv')

Y = train['Survived']

train = train.drop('Survived', axis = 1)

Num_train = train.shape[0]

Num_test = test.shape[0]

# Merging test and train datasets into gereral Data

Data = pd.concat([train, test],ignore_index=True)
Data = Data.drop('PassengerId', axis = 1)

Data['Sex']= Data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

Data['Family'] = Data['SibSp'] + Data['Parch']+1

Data = Data.drop(['Parch', 'SibSp'], axis = 1)

Data['TicketNUMB'] = np.zeros(len(Data['Ticket']))

Data['TicketNUMB'] = Data.groupby(['Ticket']).transform('count').astype(int)
# Creating a Title from Name, dropping Name

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ""



Title_Dictionary = {

                    "Capt":       5,

                    "Col":        5,

                    "Major":      5,

                    "Jonkheer":   4,

                    "Don":        4,

                    "Sir" :       4,

                    "Dr":         5,

                    "Rev":        5,

                    "Countess":   4,

                    "Dona":       4,

                    "Mme":        2,

                    "Mlle":       3,

                    "Ms":         2,

                    "Mr" :        1,

                    "Mrs" :       2,

                    "Miss" :      3,

                    "Master" :    0,

                    "Lady" :      4

                    }



Data["Title"] = Data["Name"].apply(get_title).replace(Title_Dictionary)

Data = Data.drop(['Name'], axis = 1)
# Removing NaN's in Age and mapping Age

def rm_AGE(dataset):

    age_avg = dataset.mean()

    age_std = dataset.std()

    age_null_count = dataset.isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset[np.isnan(dataset)] = age_null_random_list

    dataset = dataset.astype(int)

    dataset.loc[ dataset <= 10] = 0

    dataset.loc[(dataset > 10) & (dataset <= 20)] = 1

    dataset.loc[(dataset > 20) & (dataset <= 30)] = 2

    dataset.loc[(dataset > 30) & (dataset <= 40)] = 3

    dataset.loc[(dataset > 40) & (dataset <= 50)] = 4

    dataset.loc[(dataset > 50) & (dataset <= 60)] = 5

    dataset.loc[ dataset > 60] = 6;

    return dataset



Data['Age'] = rm_AGE(Data['Age'])
# Fill Cabin

for i in range(len(Data['Cabin'])):

    if pd.isnull(Data['Cabin'][i]):

        Data['Cabin'][i] = 0

    else:

        Data['Cabin'][i] = Data['Cabin'][i][0]

        

Data["Cabin"] = Data["Cabin"].replace({'A' : 1,

                                       'B' : 2,

                                       'C' : 3,

                                       'D' : 4,

                                       'E' : 5,

                                       'F' : 6,

                                       'G' : 7,

                                       'T' : 8})

Data["Cabin"] = Data["Cabin"].astype(int)
# Normalizing Fare by TicketNUMB



Data['Fare'] = Data['Fare'] / Data['TicketNUMB']



# Fare mean/std per Pclass, TicketNUMB, Embarked

Fare_mean = Data['Fare'].groupby([Data['Pclass'], Data['TicketNUMB'], Data['Embarked']]).mean()

Fare_std = Data['Fare'].groupby([Data['Pclass'], Data['TicketNUMB'], Data['Embarked']]).std()



# Fill NaN in Fare

for i in Data[pd.isnull(Data['Fare'])].index.tolist():

    Data['Fare'][i] = random.normalvariate(Fare_mean[Data['Pclass'][i]][int(Data['TicketNUMB'][i])][Data['Embarked'][i]], 

                                           Fare_std[Data['Pclass'][i]][int(Data['TicketNUMB'][i])][Data['Embarked'][i]])

Data['Fare'] = Data['Fare'].astype(int)    

Data = Data.drop(['Ticket', 'Embarked'], axis = 1)
Data = Data.drop(['Ticket', 'Embarked'], axis = 1)


Data.head()
Data_NP = np.array(Data)

train_NP = Data_NP[0:Num_train]

test_NP = Data_NP[Num_train:]



X_train = train_NP

Y_train = np.array(Y)

X_test = test_NP
# First look on results



# Gradient Boosting



gbc = GradientBoostingClassifier(random_state = 42)

gbc.fit(X_train, Y_train)

Y_pred = gbc.predict(X_test)

acc_gbc = round(gbc.score(X_train, Y_train), 5)

acc_gbc = np.mean(list(cross_val_score(gbc, X_train, Y_train))[0])

features_gbc = gbc.feature_importances_



# k-Nearest Neighbors



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train), 5)

acc_knn = np.mean(list(cross_val_score(knn, X_train, Y_train))[0])





# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train), 5)

acc_linear_svc = np.mean(list(cross_val_score(linear_svc, X_train, Y_train))[0])





# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred_rf = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest_re = round(random_forest.score(X_train, Y_train), 5)

acc_random_forest = np.mean(list(cross_val_score(random_forest, X_train, Y_train)))

features_rf = random_forest.feature_importances_



# AdaBoost



adaboost = AdaBoostClassifier(n_estimators=100)

adaboost.fit(X_train, Y_train)

Y_pred_adb = adaboost.predict(X_test)

adaboost.score(X_train, Y_train)

acc_adaboost_re = round(adaboost.score(X_train, Y_train), 5)

acc_adaboost = np.mean(list(cross_val_score(adaboost, X_train, Y_train)))

features_adb = adaboost.feature_importances_

# Feature importances visualization

lab = Data.columns.values

fig = plt.figure()

ax = fig.add_subplot(111)

ax.set_xticks(np.arange(0,len(lab)))

ax.set_xticklabels(Data.columns.values, rotation=315)



plt.scatter(np.arange(0,len(lab)), features_rf, color = 'red', label = u'Random Forest',)

plt.scatter(np.arange(0,len(lab)), features_gbc, color = 'blue', label = u'Gradient Boosting')

plt.scatter(np.arange(0,len(lab)), features_adb, color = 'green', label = u'AdaBoost')

plt.legend()

plt.show()

plt.close()
models = pd.DataFrame({

    'Model': ['Gradient Boosting', 'KNN', 'AdaBoost', 

              'Random Forest', 'Linear SVC'],

    'Score': [acc_gbc, acc_knn, acc_adaboost, 

              acc_random_forest,  

              acc_linear_svc]})

models.sort_values(by='Score', ascending=False)


# Random Forest 

clf_rf = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)

params_rf = {'max_depth': [None, 2, 3, 4, 5],

             'min_samples_leaf': [1, 3],

             'n_estimators': [10, 20, 30, 50, 100, 150]}



grid_rf = GridSearchCV(clf_rf, param_grid = params_rf, scoring='accuracy', cv=5)

grid_rf.fit(X_train, Y_train)



# AdaBoost 

clf_adb = AdaBoostClassifier(random_state = 42)

params_adb = {'n_estimators': [10, 20, 30, 50, 100, 150]}



grid_adb = GridSearchCV(clf_adb, param_grid=params_adb, scoring='accuracy', cv=5)

grid_adb.fit(X_train, Y_train)



# Gradient Boosting 

clf_gb = GradientBoostingClassifier(random_state=42)

params_gb = {'learning_rate': [0.01, 0.1, 0.5, 1, 10],

             'n_estimators': [50, 100, 200],

             'max_depth': [2, 3]}



grid_gb = GridSearchCV(clf_gb, param_grid=params_gb, scoring='accuracy', cv=5)

grid_gb.fit(X_train, Y_train)



# Linear SVC

clf_lsvc = LinearSVC(random_state=42)



params_lsvc = {'fit_intercept': [True, False],

               'dual': [True, False],

               'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



grid_lsvc = GridSearchCV(clf_lsvc, param_grid=params_lsvc, scoring='accuracy', cv=5)

grid_lsvc.fit(X_train, Y_train)
print(grid_rf.best_params_)

print(grid_gb.best_params_)

print(grid_lsvc.best_params_)

print(grid_adb.best_params_)

print([grid_rf.best_score_, 

       grid_gb.best_score_, 

       grid_lsvc.best_score_,

       grid_adb.best_score_])
ensemble = VotingClassifier(estimators=[

                                        ('rf', grid_rf.best_estimator_), 

                                        ('gb', grid_gb.best_estimator_), 

                                        ('lsvc', grid_lsvc.best_estimator_),

                                        ('adb', grid_adb.best_estimator_),

                                       ], voting='hard')



ensemble.fit(X_train, Y_train)



prediction = ensemble.predict(X_test) 

ensemble.score(X_train, Y_train)

acc_ensemble = round(ensemble.score(X_train, Y_train), 5)

acc_ensemble_CV = np.mean(list(cross_val_score(ensemble, X_train, Y_train)))

print(acc_ensemble)

print(acc_ensemble_CV)
#submission = pd.DataFrame({"PassengerId": PassengerId ,"Survived": prediction})

#submission.to_csv('submission.csv', index=False)