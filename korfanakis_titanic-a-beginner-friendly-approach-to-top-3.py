import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

pd.set_option('precision', 3)



import matplotlib as mpl

import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'retina'



import seaborn as sns

sns.set_style('dark')



mpl.rcParams['axes.labelsize'] = 14

mpl.rcParams['axes.titlesize'] = 15

mpl.rcParams['xtick.labelsize'] = 12

mpl.rcParams['ytick.labelsize'] = 12

mpl.rcParams['legend.fontsize'] = 12



from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler 

from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV



from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier

from xgboost import XGBClassifier



from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score



print ('Libraries Loaded!')
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')



print ('Dataframes loaded!')

print ('Training set: {} rows and {} columns'.format(train_df.shape[0], train_df.shape[1]))

print ('    Test set: {} rows and {} columns'.format(test_df.shape[0], test_df.shape[1]))
all_data = pd.concat([train_df, test_df])



print ('Combined set: {} rows and {} columns'.format(all_data.shape[0], all_data.shape[1]))

print ('\nSurvived?: ')

all_data['Survived'].value_counts(dropna = False)
train_df.head()
train_df.drop('PassengerId', axis = 1, inplace = True)
train_df.info()
missing_counts = train_df.isnull().sum().sort_values(ascending = False)

percent = (train_df.isnull().sum()*100/train_df.shape[0]).sort_values(ascending = False)



missing_df = pd.concat([missing_counts, percent], axis = 1, keys = ['Counts', '%'])

print('Missing values: ')

missing_df.head()
train_df.describe()
num_atts = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass']

train_df[num_atts].hist(figsize = (15, 6), color = 'steelblue', edgecolor = 'firebrick', linewidth = 1.5, layout = (2, 3));
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (11, 4))



sns.countplot(x = 'Sex', hue = 'Survived', data = train_df,  palette = 'tab20', ax = ax1) 

ax1.set_title('Count of (non-)Survivors by Gender')

ax1.set_xlabel('Gender')

ax1.set_ylabel('Number of Passenger')

ax1.legend(labels = ['Deceased', 'Survived'])



sns.barplot(x = 'Sex', y = 'Survived', data = train_df,  palette = ['#94BFA7', '#FFC49B'], ci = None, ax = ax2)

ax2.set_title('Survival Rate by Gender')

ax2.set_xlabel('Gender')

ax2.set_ylabel('Survival Rate');
pd.crosstab(train_df['Sex'], train_df['Survived'], normalize = 'index')
men = train_df[train_df['Sex']  == 'male']

women = train_df[train_df['Sex']  == 'female']
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (13, 4))



sns.distplot(train_df[train_df['Survived'] == 1]['Age'].dropna(), bins = 20, label = 'Survived', ax = ax1, kde = False)

sns.distplot(train_df[train_df['Survived'] == 0]['Age'].dropna(), bins = 20, label = 'Deceased', ax = ax1, kde = False)

ax1.legend()

ax1.set_title('Age Distribution - All Passengers')



sns.distplot(women[women['Survived'] == 1]['Age'].dropna(), bins = 20, label = 'Survived', ax = ax2, kde = False)

sns.distplot(women[women['Survived'] == 0]['Age'].dropna(), bins = 20, label = 'Deceased', ax = ax2, kde = False)

ax2.legend()

ax2.set_title('Age Distribution - Women')



sns.distplot(men[men['Survived'] == 1]['Age'].dropna(), bins = 20, label = 'Survived', ax = ax3, kde = False)

sns.distplot(men[men['Survived'] == 0]['Age'].dropna(), bins = 20, label = 'Deceased', ax = ax3, kde = False)

ax3.legend()

ax3.set_title('Age Distribution - Men')



plt.tight_layout();
# train_df['Age_Bin'] = pd.qcut(train_df['Age'], 4)  # Quantile-based discretization

train_df['Age_Bin'] = (train_df['Age']//15)*15

train_df[['Age_Bin', 'Survived']].groupby(['Age_Bin']).mean()
sns.countplot(x = 'Embarked', hue = 'Survived', data = train_df,  palette = 'tab20') 

plt.ylabel('Number of Passenger')

plt.title('Count of (non-)Survivors by Port of Embarkation')

plt.legend(['Deceased', 'Survived']);
print ('Number of passengers in each class:')

train_df['Pclass'].value_counts()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))



sns.countplot(x = 'Pclass', hue = 'Survived', data = train_df,  palette = 'tab20', ax = ax1) 

ax1.legend(['Deceased', 'Survived'])

ax1.set_title('Count of (non-)Survivors by Class')

ax1.set_ylabel('Number of Passengers')



sns.barplot(x = 'Pclass', y = 'Survived', data = train_df,  palette = ['#C98BB9', '#F7D4BC', '#B5E2FA'], ci = None, ax = ax2)

ax2.set_title('Survival Rate by Class')

ax2.set_ylabel('Survival Rate');
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))



sns.boxplot(x = 'Pclass', y = 'Fare', data = train_df, palette = 'tab20', ax = ax1)

ax1.set_title('Distribution of Fares by Class')



sns.distplot(train_df[train_df['Survived'] == 1]['Fare'], label = 'Survived', ax = ax2)

sns.distplot(train_df[train_df['Survived'] == 0]['Fare'], label = 'Not Survived', ax = ax2)

ax2.set_title('Distribution of Fares for (non-)Survivors')

ax2.set_xlim([-20, 200])

ax2.legend();
train_df['Fare_Bin'] = pd.qcut(train_df['Fare'], 5)

train_df[['Fare_Bin', 'Survived']].groupby(['Fare_Bin']).mean()
alone = train_df[(train_df['SibSp'] == 0) & (train_df['Parch'] == 0)]

not_alone = train_df[(train_df['SibSp'] != 0) | (train_df['Parch'] != 0)]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))



sns.countplot(x = 'Survived', data = alone,  palette = 'tab20', ax = ax1) 

ax1.set_title('Count of Alone (non-)Survivors')

ax1.set_xlabel('')

ax1.set_xticklabels(['Deceased', 'Survived'])

ax1.set_ylabel('Number of Passengers')



sns.countplot(x = 'Survived', data = not_alone,  palette = 'tab20', ax = ax2) 

ax2.set_title('Count of (non-)Survivors with Family Onboard')

ax2.set_xlabel('')

ax2.set_xticklabels(['Deceased', 'Survived'])

ax2.set_ylabel('Number of Passengers')



plt.tight_layout();
train_df['Relatives'] = train_df['SibSp'] + train_df['Parch']

# train_df[['Relatives', 'Survived']].groupby(['Relatives']).mean()



sns.factorplot('Relatives', 'Survived', data = train_df, color = 'firebrick', aspect = 1.5)

plt.title('Survival rate by Number of Relatives Onboard');
train_df['Title'] = train_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())



train_df['Title'].replace({'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss'}, inplace = True)

train_df['Title'].replace(['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'the Countess', 'Jonkheer'],

                           'Rare Title', inplace = True)

train_df['Title'].value_counts()
cols = ['#067BC2', '#84BCDA', '#ECC30B', '#F37748', '#D56062']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 4))



sns.countplot(x = 'Title', data = train_df,  palette = cols, ax = ax1)

ax1.set_title('Passenger Count by Title')

ax1.set_ylabel('Number of Passengers')



sns.barplot(x = 'Title', y = 'Survived', data = train_df,  palette = cols, ci = None, ax = ax2)

ax2.set_title('Survival Rate by Title')

ax2.set_ylabel('Survival Rate');
print ('Cabin:\n  Number of existing values: ', train_df['Cabin'].notnull().sum())

print ('    Number of unique values: ', train_df['Cabin'].nunique())
all_data['Age'] = all_data['Age'].fillna(train_df['Age'].median())

all_data['Fare'] = all_data['Fare'].fillna(train_df['Fare'].median())

print ('Done!')
# Again, the code for 'Family_Survival' comes from this kernel:

# https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83/notebook



all_data['Last_Name'] = all_data['Name'].apply(lambda x: str.split(x, ',')[0])

all_data['Fare'].fillna(all_data['Fare'].mean(), inplace = True)



default_sr_value = 0.5

all_data['Family_Survival'] = default_sr_value



for grp, grp_df in all_data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId', 'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

    

    if (len(grp_df) != 1):  # A Family group is found.

        for ind, row in grp_df.iterrows():

            smax = grp_df.drop(ind)['Survived'].max()

            smin = grp_df.drop(ind)['Survived'].min()

            passID = row['PassengerId']

            

            if (smax == 1.0):

                all_data.loc[all_data['PassengerId'] == passID, 'Family_Survival'] = 1

            elif (smin==0.0):

                all_data.loc[all_data['PassengerId'] == passID, 'Family_Survival'] = 0



for _, grp_df in all_data.groupby('Ticket'):

    

    if (len(grp_df) != 1):

        for ind, row in grp_df.iterrows():

            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):

                smax = grp_df.drop(ind)['Survived'].max()

                smin = grp_df.drop(ind)['Survived'].min()

                passID = row['PassengerId']

                

                if (smax == 1.0):

                    all_data.loc[all_data['PassengerId'] == passID, 'Family_Survival'] = 1

                elif (smin==0.0):

                    all_data.loc[all_data['PassengerId'] == passID, 'Family_Survival'] = 0

                    

#####################################################################################

all_data['Age_Bin'] = (all_data['Age']//15)*15

all_data['Fare_Bin'] = pd.qcut(all_data['Fare'], 5)

all_data['Relatives'] = all_data['SibSp'] + all_data['Parch']

#####################################################################################

all_data['Title'] = all_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

all_data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'}, inplace = True)

all_data['Title'].replace(['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'the Countess', 'Jonkheer', 'Dona'],

                           'Rare Title', inplace = True)    



print ('Done!')
all_data['Fare_Bin'] = LabelEncoder().fit_transform(all_data['Fare_Bin'])

all_data['Age_Bin'] = LabelEncoder().fit_transform(all_data['Age_Bin'])

all_data['Title_Bin'] = LabelEncoder().fit_transform(all_data['Title'])

all_data['Sex'] = LabelEncoder().fit_transform(all_data['Sex'])



print ('Done!')
all_data.drop(['PassengerId', 'Age', 'Fare', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Title', 'Last_Name', 'Embarked'], axis = 1, inplace = True)



print ('Done!')

print ('Modified dataset: ')

all_data.head()
train_df = all_data[:891]



X_train = train_df.drop('Survived', 1)

y_train = train_df['Survived']



#######################################################



test_df = all_data[891:]



X_test = test_df.copy()

X_test.drop('Survived', axis = 1, inplace = True)

print ('Splitting: Done!')
std_scaler = StandardScaler()



X_train_scaled = std_scaler.fit_transform(X_train)  # fit_transform the X_train

X_test_scaled = std_scaler.transform(X_test)        # only transform the X_test



print ('Scaling: Done!')
random_state = 1



# Step 1: create a list containing all estimators with their default parameters

clf_list = [GaussianNB(), 

            LogisticRegression(random_state = random_state),

            KNeighborsClassifier(), 

            SVC(random_state = random_state, probability = True),

            DecisionTreeClassifier(random_state = random_state), 

            RandomForestClassifier(random_state = random_state),

            XGBClassifier(random_state = random_state), 

            AdaBoostClassifier(base_estimator = DecisionTreeClassifier(random_state = random_state), random_state = random_state)]





# Step 2: calculate the cv mean and standard deviation for each one of them

cv_base_mean, cv_std = [], []

for clf in clf_list:  

    

    cv = cross_val_score(clf, X_train_scaled, y = y_train, scoring = 'accuracy', cv = 5, n_jobs = -1)

    

    cv_base_mean.append(cv.mean())

    cv_std.append(cv.std())



    

# Step 3: create a dataframe and plot the mean with error bars

cv_total = pd.DataFrame({'Algorithm': ['Gaussian Naive Bayes', 'Logistic Regression', 'k-Nearest Neighboors', 'SVC', 'Decision Tree', 'Random Forest', 'XGB Classifier', 'AdaBoost Classifier'],

                         'CV-Means': cv_base_mean, 

                         'CV-Errors': cv_std})



sns.barplot('CV-Means', 'Algorithm', data = cv_total, palette = 'Paired', orient = 'h', **{'xerr': cv_std})

plt.xlabel('Mean Accuracy')

plt.title('Cross Validation Scores')

plt.xlim([0.725, 0.88])

plt.axvline(x = 0.80, color = 'firebrick', linestyle = '--');
estimators = [('gnb', clf_list[0]), ('lr', clf_list[1]),

              ('knn', clf_list[2]), ('svc', clf_list[3]),

              ('dt', clf_list[4]), ('rf', clf_list[5]),

              ('xgb', clf_list[6]), ('ada', clf_list[7])]



base_voting_hard = VotingClassifier(estimators = estimators , voting = 'hard')

base_voting_soft = VotingClassifier(estimators = estimators , voting = 'soft') 



cv_hard = cross_val_score(base_voting_hard, X_train_scaled, y_train, cv = 5)

cv_soft = cross_val_score(base_voting_soft, X_train_scaled, y_train, cv = 5)



print ('Baseline Models - Ensemble\n--------------------------')

print ('Hard Voting: {}%'.format(np.round(cv_hard.mean()*100, 1)))

print ('Soft Voting: {}%'.format(np.round(cv_soft.mean()*100, 1)))
base_voting_hard.fit(X_train_scaled, y_train)

base_voting_soft.fit(X_train_scaled, y_train)



y_pred_base_hard = base_voting_hard.predict(X_test_scaled)

y_pred_base_soft = base_voting_hard.predict(X_test_scaled)
cv_means_tuned = [np.nan] # we can't actually tune the GNB classifier, so we fill its element with NaN



#simple performance reporting function

def clf_performance(classifier, model_name):

    print(model_name)

    print('-------------------------------')

    print('   Best Score: ' + str(classifier.best_score_))

    print('   Best Parameters: ' + str(classifier.best_params_))

    

    cv_means_tuned.append(classifier.best_score_)
lr = LogisticRegression()



param_grid = {'max_iter' : [100],

              'penalty' : ['l1', 'l2'],

              'C' : np.logspace(-2, 2, 20),

              'solver' : ['lbfgs', 'liblinear']}



clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)



best_clf_lr = clf_lr.fit(X_train_scaled, y_train)

clf_performance(best_clf_lr, 'Logistic Regression')
# n_neighbors = np.concatenate((np.arange(3, 30, 1), np.arange(22, 32, 2)))



knn = KNeighborsClassifier()

param_grid = {'n_neighbors' : np.arange(3, 30, 2),

              'weights': ['uniform', 'distance'],

              'algorithm': ['auto'],

              'p': [1, 2]}



clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)

best_clf_knn = clf_knn.fit(X_train_scaled, y_train)

clf_performance(best_clf_knn, 'KNN')
svc = SVC(probability = True)

param_grid = tuned_parameters = [{'kernel': ['rbf'], 

                                  'gamma': [0.01, 0.1, 0.5, 1, 2, 5],

                                  'C': [.1, 1, 2, 5]},

                                 {'kernel': ['linear'], 

                                  'C': [.1, 1, 2, 10]},

                                 {'kernel': ['poly'], 

                                  'degree' : [2, 3, 4, 5], 

                                  'C': [.1, 1, 10]}]



clf_svc = GridSearchCV(svc, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)

best_clf_svc = clf_svc.fit(X_train_scaled, y_train)

clf_performance(best_clf_svc, 'SVC')
dt = DecisionTreeClassifier(random_state = 1)

param_grid = {'max_depth': [3, 5, 10, 20, 50],

              'criterion': ['entropy', 'gini'],

              'min_samples_split': [5, 10, 15, 30],

              'max_features': [None, 'auto', 'sqrt', 'log2']}

                                  

clf_dt = GridSearchCV(dt, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)

best_clf_dt = clf_dt.fit(X_train_scaled, y_train)

clf_performance(best_clf_dt, 'Decision Tree')
rf = RandomForestClassifier(random_state = 42)

param_grid = {'n_estimators': [50, 150, 300, 450],

              'criterion': ['entropy'],

              'bootstrap': [True],

              'max_depth': [3, 5, 10],

              'max_features': ['auto','sqrt'],

              'min_samples_leaf': [2, 3],

              'min_samples_split': [2, 3]}

                                  

clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)

best_clf_rf = clf_rf.fit(X_train_scaled, y_train)

clf_performance(best_clf_rf, 'Random Forest')
best_rf = best_clf_rf.best_estimator_



importances = pd.DataFrame({'Feature': X_train.columns,

                            'Importance': np.round(best_rf.feature_importances_, 3)})



importances = importances.sort_values('Importance', ascending = True).set_index('Feature')



importances.plot.barh(color = 'steelblue', edgecolor = 'firebrick', legend=False)

plt.title('Random Forest Classifier')

plt.xlabel('Importance');
xgb = XGBClassifier(random_state = 42)



param_grid = {'n_estimators': [15, 25, 50, 100],

              'colsample_bytree': [0.65, 0.75, 0.80],

              'max_depth': [None],

              'reg_alpha': [1],

              'reg_lambda': [1, 2, 5],

              'subsample': [0.50, 0.75, 1.00],

              'learning_rate': [0.01, 0.1, 0.5],

              'gamma': [0.5, 1, 2, 5],

              'min_child_weight': [0.01],

              'sampling_method': ['uniform']}



clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 3, verbose = True, n_jobs = -1)

best_clf_xgb = clf_xgb.fit(X_train_scaled, y_train)

clf_performance(best_clf_xgb, 'XGB')
best_xgb = best_clf_xgb.best_estimator_



importances = pd.DataFrame({'Feature': X_train.columns,

                            'Importance': np.round(best_xgb.feature_importances_, 3)})



importances = importances.sort_values('Importance', ascending = True).set_index('Feature')



importances.plot.barh(color = 'darkgray', edgecolor = 'firebrick', legend = False)

plt.title('XGBoost Classifier')

plt.xlabel('Importance');
adaDTC = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(random_state = random_state), random_state=random_state)



param_grid = {'algorithm': ['SAMME', 'SAMME.R'],

              'base_estimator__criterion' : ['gini', 'entropy'],

              'base_estimator__splitter' : ['best', 'random'],

              'n_estimators': [2, 5, 10, 50],

              'learning_rate': [0.01, 0.1, 0.2, 0.3, 1, 2]}



clf_ada = GridSearchCV(adaDTC, param_grid = param_grid, cv = 5, scoring = 'accuracy', n_jobs = -1, verbose = 1)

best_clf_ada = clf_ada.fit(X_train_scaled, y_train)



clf_performance(best_clf_ada, 'AdaBost')
best_ada = best_clf_ada.best_estimator_

importances = pd.DataFrame({'Feature': X_train.columns,

                            'Importance': np.round(best_ada.feature_importances_, 3)})



importances = importances.sort_values('Importance', ascending = True).set_index('Feature')



importances.plot.barh(color = 'cadetblue', edgecolor = 'firebrick', legend = False)

plt.title('AdaBoost Classifier')

plt.xlabel('Importance');
cv_total = pd.DataFrame({'Algorithm': ['Gaussian Naive Bayes', 'Logistic Regression', 'k-Nearest Neighboors', 'SVC', 'Decision Tree', 'Random Forest', 'XGB Classifier', 'AdaBoost Classifier'],

                         'Baseline': cv_base_mean, 

                         'Tuned Performance': cv_means_tuned})



cv_total
best_lr = best_clf_lr.best_estimator_

best_knn = best_clf_knn.best_estimator_

best_svc = best_clf_svc.best_estimator_

best_dt = best_clf_dt.best_estimator_

best_rf = best_clf_rf.best_estimator_

best_xgb = best_clf_xgb.best_estimator_

# best_ada = best_clf_ada.best_estimator_  # didn't help me in my final ensemble



estimators = [('lr', best_lr), ('knn', best_knn), ('svc', best_svc),

              ('rf', best_rf), ('xgb', best_xgb), ('dt', best_dt)]



tuned_voting_hard = VotingClassifier(estimators = estimators, voting = 'hard', n_jobs = -1)

tuned_voting_soft = VotingClassifier(estimators = estimators, voting = 'soft', n_jobs = -1)



tuned_voting_hard.fit(X_train_scaled, y_train)

tuned_voting_soft.fit(X_train_scaled, y_train)



cv_hard = cross_val_score(tuned_voting_hard, X_train_scaled, y_train, cv = 5)

cv_soft = cross_val_score(tuned_voting_soft, X_train_scaled, y_train, cv = 5)



print ('Tuned Models - Ensemble\n-----------------------')

print ('Hard Voting: {}%'.format(np.round(cv_hard.mean()*100, 2)))

print ('Soft Voting: {}%'.format(np.round(cv_soft.mean()*100, 2)))



y_pred_tuned_hd = tuned_voting_hard.predict(X_test_scaled).astype(int)

y_pred_tuned_sf = tuned_voting_soft.predict(X_test_scaled).astype(int)
test_df = pd.DataFrame(pd.read_csv('../input/titanic/test.csv')['PassengerId'])



pd.DataFrame(data = {'PassengerId': test_df.PassengerId, 

                     'Survived': y_pred_base_hard.astype(int)}).to_csv('01-Baseline_Hard_voting.csv', index = False)



pd.DataFrame(data = {'PassengerId': test_df.PassengerId, 

                     'Survived': y_pred_base_soft.astype(int)}).to_csv('02-Baseline_Soft_voting.csv', index = False)



pd.DataFrame(data = {'PassengerId': test_df.PassengerId, 

                     'Survived': y_pred_tuned_hd.astype(int)}).to_csv('03-Tuned_Hard_Voting.csv', index = False)



pd.DataFrame(data = {'PassengerId': test_df.PassengerId, 

                     'Survived': y_pred_tuned_sf.astype(int)}).to_csv('04-Tuned_Soft_Voting.csv', index = False)