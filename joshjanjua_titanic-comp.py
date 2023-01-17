# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#Machine Learning

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score

from sklearn.metrics import classification_report, accuracy_score, precision_score, confusion_matrix, roc_curve, auc

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.naive_bayes import GaussianNB



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



#format columns/rows

pd.set_option('display.max_columns', None)

pd.options.display.float_format = '{:.4f}'.format 

# Hide system warnings

import sys

import warnings



if not sys.warnoptions:

    warnings.simplefilter("ignore")

    

# Any results you write to the current directory are saved as output.
test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')

test.shape,train.shape
train.info()
train.describe(include = 'all')
train.head()
#training set missing most cabin and some age

sns.heatmap(train.isnull())

plt.title('Missing Train Data \n')
#Also some embarked

train.isnull().sum()
#test looks to be the same

sns.heatmap(test.isnull(),cmap = 'coolwarm')

plt.title('Missing Test Data \n')
#No embarked, but 1 fare missing

test.isnull().sum()
#Deeper analysis needed for cabin since it's above 40% missing in both data sets

print('Train','\n',(train.isnull().sum()/len(train))*100)

print('-'*20)

print('Test','\n',(test.isnull().sum()/len(test))*100)
#Looks to be different ages by Pclass

sns.boxplot(x='Pclass',y='Age',data=train)

plt.title('Age by Class \n')
#confirmed use these to input missing ages

train.groupby(['Pclass'], as_index = False)['Age'].median()
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]  

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(lambda x: impute_age(x), axis = 1)

test['Age'] = test[['Age','Pclass']].apply(lambda x: impute_age(x), axis = 1)
#Imput S since it is the mode

train['Embarked'].value_counts()
train['Embarked'] = np.where(~train['Embarked'].isnull(),train['Embarked'],'S')
#Use median since Fare is so skewed

sns.distplot(test['Fare'],bins=50)

plt.title('Distribution of Fare \n')

#use median 

round(test['Fare'].mean(),4),test['Fare'].median()
test['Fare'] = np.where(~test['Fare'].isnull(),test['Fare'],test['Fare'].median())
#Resolved all missing except Cabin

print('Train','\n',(train.isnull().sum()/len(train))*100)

print('-'*20)

print('Test','\n',(test.isnull().sum()/len(test))*100)
#Cabin Analysis - read some mention a cabin known variable was helpful - seems if you're cabin was known you were more likely to survive

train['Cabinknown'] = 0

train['Cabinknown'] = np.where((train['Cabin'].isnull()),train['Cabinknown'],1)

test['Cabinknown'] = 0

test['Cabinknown'] = np.where((test['Cabin'].isnull()),test['Cabinknown'],1)



train[['Cabinknown','Survived']].groupby('Cabinknown').mean()
#Also read that pulling the cabin letter may be helpful as well

#anything you can pull from Cabin - yes, use the letter

c = train[~train['Cabin'].isnull()]['Cabin']

c.head()
train['Cabinletter'] = train['Cabin'].str[0]

test['Cabinletter'] = test['Cabin'].str[0]
#From my MOOC, I remember grabbing title was a good predictor so let's do that

train['Title'] = train['Name'].str.split(r'\s*,\s*|\s*\.\s*').str[1]

test['Title'] = test['Name'].str.split(r'\s*,\s*|\s*\.\s*').str[1]
#From previous analysis, I identified that SibSp and Parch are weak predictors individually and have similar distributions so I am combining

train['Family'] = train['SibSp'] + train['Parch'] + 1

test['Family'] = test['SibSp'] + test['Parch'] + 1
#looks like kids and elderly survived at a higher rate than did others

plt.figure(figsize=(10,6)) 

train[train['Survived'] ==0]['Age'].hist(alpha=.7, bins=30, label='Survived =0') 

train[train['Survived'] ==1]['Age'].hist(alpha=.7, bins=30, label='Survived =1') 

plt.legend(loc='upper right') 

plt.title('Age Survival \n')

plt.show() 
train['Agegroup'] = np.where(train['Age']<17,0,(np.where((train['Age']>17) & (train['Age']<55),1,2)))

test['Agegroup'] = np.where(test['Age']<17,0,(np.where((test['Age']>17) & (test['Age']<55),1,2)))
round(train['Fare'].mean(),4),train['Fare'].median()
#Looks like those who paid more survived at a higher rate

plt.figure(figsize=(10,6)) 

train[train['Survived'] ==0]['Fare'].hist(alpha=.7, bins=30, label='Survived =0') 

train[train['Survived'] ==1]['Fare'].hist(alpha=.7, bins=50, label='Survived =1') 

plt.legend(loc='upper right') 

plt.title('Fare Survival \n')

plt.show() 
train['Faregroup'] = np.where(train['Fare']<15,0,1)

test['Faregroup'] = np.where(test['Fare']<15,0,1)
#Looks like traditional families had a higher survival rate than individuals or large families 

plt.figure(figsize=(10,6)) 

train[train['Survived'] ==0]['Family'].hist(alpha=.7, bins=30, label='Survived =0') 

train[train['Survived'] ==1]['Family'].hist(alpha=.7, bins=30, label='Survived =1') 

plt.legend(loc='upper right') 

plt.title('Family Survival \n')

plt.show() 
train['Famgroup'] = np.where(train['Family']<2,0,(np.where((train['Family']>1) & (train['Family']<5),1,2)))

test['Famgroup'] = np.where(test['Age']<2,0,(np.where((test['Family']>1) & (test['Family']<5),1,2)))
#drop unnecessary columns - either wrong type or accounted for elsewhere

traina = train.drop(columns=['Name','Ticket','Cabin','Fare','Age','Family','SibSp','Parch'])

testa = test.drop(columns=['Name','Ticket','Cabin','Fare','Age','Family','SibSp','Parch'])
#doesnt look like any of the variables are too highly correlated

plt.figure(figsize = (10,10))

sns.heatmap(traina.corr(), annot = True)

plt.title('Variable Heatmap \n')
#store test ID for submission purposes

TestId=testa['PassengerId']

#align data set shapes and get dummies

total_features=pd.concat((traina.drop(['PassengerId','Survived'], axis=1), testa.drop(['PassengerId'], axis=1)))

total_features=pd.get_dummies(total_features, drop_first=True)

train_features=total_features[0:traina.shape[0]]



#making sure the test set matches the train set

test_features=total_features[traina.shape[0]:] 
train_features.shape,test_features.shape
X = train_features

y = traina['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
ran = RandomForestClassifier(random_state=1)

knn = KNeighborsClassifier()

log = LogisticRegression()

xgb = XGBClassifier()

gbc = GradientBoostingClassifier()

svc = SVC(probability=True)

ext = ExtraTreesClassifier()

ada = AdaBoostClassifier()

gnb = GaussianNB()

gpc = GaussianProcessClassifier()

bag = BaggingClassifier()



# Prepare lists

models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]         

scores = []



# Sequentially fit and cross validate all models

for mod in models:

    mod.fit(X_train, y_train)

    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 10)

    scores.append(acc.mean())
# Creating a table of results, ranked highest to lowest

results = pd.DataFrame({

    'Model': ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression', 'XGBoost', 'Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost', 'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier'],

    'Score': scores})



result_df = results.sort_values(by='Score', ascending=False).reset_index(drop=True)

result_df
sns.barplot(x='Score', y='Model',data = result_df, color = 'skyblue')

plt.title('Model Accuracy \n')

plt.xlim(0.65, 0.85)
# Getting feature importances for the 5 models where we can

xgb_imp = pd.DataFrame({'Feature':train_features.columns, 'xgb importance':xgb.feature_importances_})

gbc_imp = pd.DataFrame({'Feature':train_features.columns, 'gbc importance':gbc.feature_importances_})

ran_imp = pd.DataFrame({'Feature':train_features.columns, 'ran importance':ran.feature_importances_})

ext_imp = pd.DataFrame({'Feature':train_features.columns, 'ext importance':ext.feature_importances_})

ada_imp = pd.DataFrame({'Feature':train_features.columns, 'ada importance':ada.feature_importances_})



# Merging results into a single dataframe

importances = gbc_imp.merge(xgb_imp, on='Feature').merge(ran_imp, on='Feature').merge(ext_imp, on='Feature').merge(ada_imp, on='Feature')



# Calculating average importance per feature

importances['Average'] = importances.mean(axis=1)



# Ranking top to bottom

importances = importances.sort_values(by='Average', ascending=False).reset_index(drop=True)



# Display

importances
plt.figure(figsize = (7,7))

sns.barplot(y = 'Feature', x = 'Average',data = importances, color = 'skyblue')

plt.title('Feature Importances \n')
#dropping unimportant features

columns = importances[importances['Average']<.03]['Feature'].values

train_features.drop(columns = columns, inplace = True)

test_features.drop(columns = columns, inplace = True)
X = train_features

y = traina['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
# Initiate models

ran = RandomForestClassifier(random_state=1)

knn = KNeighborsClassifier()

log = LogisticRegression()

xgb = XGBClassifier(random_state=1)

gbc = GradientBoostingClassifier(random_state=1)

svc = SVC(probability=True)

ext = ExtraTreesClassifier(random_state=1)

ada = AdaBoostClassifier(random_state=1)

gnb = GaussianNB()

gpc = GaussianProcessClassifier()

bag = BaggingClassifier(random_state=1)



# Lists

models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]         

scores_v2 = []



# Fit & cross validate

for mod in models:

    mod.fit(X_train, y_train)

    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 10)

    scores_v2.append(acc.mean())
# Creating a table of results, ranked highest to lowest

results = pd.DataFrame({

    'Model': ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression', 'XGBoost', 'Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost', 'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier'],

    'Score': scores,

    'Score w/Feature Selection': scores_v2})



result_df = results.sort_values(by='Score w/Feature Selection', ascending=False).reset_index(drop=True)

result_df
# Parameter's to search

penalty = ['l1', 'l2']

C = np.logspace(0, 4, 10)



# Setting up parameter grid

hyperparams = {'penalty': penalty, 'C': C}



# Run GridSearch CV

lrgd=GridSearchCV(estimator = LogisticRegression(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy", n_jobs =-1)



# Fitting model and return results

lrgd.fit(X_train, y_train)

print(lrgd.best_score_)

print(lrgd.best_estimator_)
# Parameter's to search

n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]

algorithm = ['auto']

weights = ['uniform', 'distance']

leaf_size = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]



# Setting up parameter grid

hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 

               'n_neighbors': n_neighbors}



# Run GridSearch CV

kngd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy", n_jobs =-1)



# Fitting model and return results

kngd.fit(X_train, y_train)

print(kngd.best_score_)

print(kngd.best_estimator_)
# Parameter's to search

n_estimators = [10, 25, 50, 75, 100]

max_depth = [3, None]

max_features = [1, 3, 5, 7]

min_samples_split = [2, 4, 6, 8, 10]

min_samples_leaf = [2, 4, 6, 8, 10]



# Setting up parameter grid

hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features,

               'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}



# Run GridSearch CV

rfgd=GridSearchCV(estimator = RandomForestClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy", n_jobs =-1)



# Fitting model and return results

rfgd.fit(X_train, y_train)

print(rfgd.best_score_)

print(rfgd.best_estimator_)
# Parameter's to search

Cs = [0.001, 0.01, 0.1, 1, 5, 10, 15, 20, 50, 100]

gammas = [0.001, 0.01, 0.1, 1]



# Setting up parameter grid

hyperparams = {'C': Cs, 'gamma' : gammas}



# Run GridSearch CV

svgd=GridSearchCV(estimator = SVC(probability=True), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy", n_jobs =-1)



# Fitting model and return results

svgd.fit(X_train, y_train)

print(svgd.best_score_)

print(svgd.best_estimator_)
# Parameter's to search

n_restarts_optimizer = [0, 1, 2, 3]

max_iter_predict = [1, 2, 5, 10, 20, 35, 50, 100]

warm_start = [True, False]



# Setting up parameter grid

hyperparams = {'n_restarts_optimizer': n_restarts_optimizer, 'max_iter_predict': max_iter_predict, 'warm_start': warm_start}



# Run GridSearch CV

gpgd=GridSearchCV(estimator = GaussianProcessClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy", n_jobs =-1)



# Fitting model and return results

gpgd.fit(X_train, y_train)

print(gpgd.best_score_)

print(gpgd.best_estimator_)
# Parameter's to search

n_estimators = [10, 25, 50, 75, 100, 125, 150, 200]

learning_rate = [0.001, 0.01, 0.1, 0.5, 1, 1.5, 2]



# Setting up parameter grid

hyperparams = {'n_estimators': n_estimators, 'learning_rate': learning_rate}



# Run GridSearch CV

adgd=GridSearchCV(estimator = AdaBoostClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy", n_jobs =-1)



# Fitting model and return results

adgd.fit(X_train, y_train)

print(adgd.best_score_)

print(adgd.best_estimator_)
# Parameter's to search

learning_rate = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

n_estimators = [100, 250, 500, 750, 1000, 1250, 1500]



# Setting up parameter grid

hyperparams = {'learning_rate': learning_rate, 'n_estimators': n_estimators}



# Run GridSearch CV

gbgd=GridSearchCV(estimator = GradientBoostingClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy", n_jobs =-1)



# Fitting model and return results

gbgd.fit(X_train, y_train)

print(gbgd.best_score_)

print(gbgd.best_estimator_)
# Parameter's to search

n_estimators = [10, 25, 50, 75, 100]

max_depth = [3, None]

max_features = [1, 3, 5, 7]

min_samples_split = [2, 4, 6, 8, 10]

min_samples_leaf = [2, 4, 6, 8, 10]



# Setting up parameter grid

hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features,

               'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}



# Run GridSearch CV

etgd=GridSearchCV(estimator = ExtraTreesClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy", n_jobs =-1)



# Fitting model and return results

etgd.fit(X_train, y_train)

print(etgd.best_score_)

print(etgd.best_estimator_)
# Parameter's to search

n_estimators = [10, 15, 20, 25, 50, 75, 100, 150]

max_samples = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 50]

max_features = [1, 3, 5, 7]



# Setting up parameter grid

hyperparams = {'n_estimators': n_estimators, 'max_samples': max_samples, 'max_features': max_features}



# Run GridSearch CV

bcgd=GridSearchCV(estimator = BaggingClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy", n_jobs =-1)



# Fitting model and return results

bcgd.fit(X_train, y_train)

print(bcgd.best_score_)

print(bcgd.best_estimator_)
# Parameter's to search

learning_rate = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

n_estimators = [10, 25, 50, 75, 100, 250, 500, 750, 1000]



# Setting up parameter grid

hyperparams = {'learning_rate': learning_rate, 'n_estimators': n_estimators}



# Run GridSearch CV

gd=GridSearchCV(estimator = XGBClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy", n_jobs=-1)



# Fitting model and return results

gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
max_depth = [3, 4, 5, 6, 7, 8, 9, 10]

min_child_weight = [1, 2, 3, 4, 5, 6]



hyperparams = {'max_depth': max_depth, 'min_child_weight': min_child_weight}



gd=GridSearchCV(estimator = gd.best_estimator_, param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy",n_jobs=-1)



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
gamma = [i*0.1 for i in range(0,5)]



hyperparams = {'gamma': gamma}



gd=GridSearchCV(estimator = gd.best_estimator_,param_grid = hyperparams, verbose=True, cv=5, scoring = "accuracy",n_jobs=-1)



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
subsample = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

colsample_bytree = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

    

hyperparams = {'subsample': subsample, 'colsample_bytree': colsample_bytree}



gd=GridSearchCV(estimator = gd.best_estimator_, param_grid = hyperparams, verbose=True, cv=5, scoring = "accuracy",n_jobs=-1)



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
reg_alpha = [1e-5, 1e-2, 0.1, 1, 100]

    

hyperparams = {'reg_alpha': reg_alpha}



xggd=GridSearchCV(estimator = gd.best_estimator_,param_grid = hyperparams, verbose=True, cv=5, scoring = "accuracy",n_jobs=-1)



xggd.fit(X_train, y_train)

print(xggd.best_score_)

print(xggd.best_estimator_)
# Initiate tuned models 

ran = rfgd.best_estimator_



knn = kngd.best_estimator_



log = lrgd.best_estimator_



xgb = xggd.best_estimator_

                    

gbc = gbgd.best_estimator_



svc = svgd.best_estimator_



ext = etgd.best_estimator_



ada = adgd.best_estimator_



gpc = gpgd.best_estimator_



bag = bcgd.best_estimator_



# Lists

models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]         

scores_v3 = []



# Fit & cross-validate

for mod in models:

    mod.fit(X_train, y_train)

    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 10)

    scores_v3.append(acc.mean())
# Creating a table of results, ranked highest to lowest

results = pd.DataFrame({

    'Model': ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression', 'XGBoost', 'Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost', 'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier'],

    'Original Score': scores,

    'Score with feature selection': scores_v2,

    'Score with tuned parameters': scores_v3})



result_df = results.sort_values(by='Score with tuned parameters', ascending=False).reset_index(drop=True)

result_df
# Initiate tuned models 

ran = rfgd.best_estimator_



knn = kngd.best_estimator_



log = lrgd.best_estimator_



xgb = xggd.best_estimator_

                    

gbc = gbgd.best_estimator_



svc = svgd.best_estimator_



ext = etgd.best_estimator_



ada = adgd.best_estimator_



gpc = gpgd.best_estimator_



bag = bcgd.best_estimator_



# Lists

models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]         

scores_v4 = []



# Fit & cross-validate

for mod in models:

    mod.fit(X_train, y_train)

    predict = mod.predict(X_test)

    acc = accuracy_score(y_test, predict)

    scores_v4.append(acc)
# Creating a table of results, ranked highest to lowest

results = pd.DataFrame({

    'Model': ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression', 'XGBoost', 'Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost', 'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier'],

    'Original Score': scores,

    'Score with feature selection': scores_v2,

    'Score with tuned parameters': scores_v3,

    'Score with full accuracy': scores_v4})



result_df = results.sort_values(by='Score with full accuracy', ascending=False).reset_index(drop=True)

result_df
predictions = ext.predict(test_features)
#test shape 

TestId.shape,predictions.shape
#Create submission (if competition)

submission=pd.DataFrame()

submission['PassengerId']=TestId

submission['Survived']=predictions

submission.to_csv('submission.csv', index=False)