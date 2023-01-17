# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import re

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
full_df = train.append(test, ignore_index=True, sort=False)
full_df.info()
full_df.isnull().sum()
def name_split(row):

    name=row['Name']

    maiden_regex = re.compile('(\([a-zA-Z\s]+\))') #regex to select any character between parenthesis

    maiden_match = maiden_regex.search(name)

    nick_regex = re.compile('(\"[a-zA-Z\s]+\")') #regex to select any character between quote character

    nick_match = nick_regex.search(name)

    

    if maiden_match:

        maiden_name = maiden_match.groups()[0]

        name = name.replace(maiden_name, '')

        maiden_name = maiden_name.strip('()')

    else:

        maiden_name = np.nan

        

    if nick_match:

        nick_name = nick_match.groups()[0]

        name = name.replace(nick_name, '')

        nick_name = nick_name.strip('"')

    else:

        nick_name = np.nan

    

    #Additionnal strip is added to remove every unnecessary white space

    

    last_name = name.split(',')[0].strip()

    

    title = name.split(',')[1].split('.')[0].strip()

    

    first_name = name.split(',')[1].split('.')[1].strip()

        

    return (title, last_name, first_name, maiden_name, nick_name)
splitted_name = full_df.apply(name_split, axis=1, result_type='expand') # Use of apply on the row instead of map to be able to use 'expand'

splitted_name.columns=['Title','LastName', 'FirstName', 'MaidenName', 'NickName']

full_df = full_df.join(splitted_name)

full_df.drop('Name', axis=1, inplace=True)
{feature:len(full_df[feature].unique()) for feature in splitted_name.columns}





full_df['Title'].value_counts()
titles = full_df['Title'].unique().tolist()

titles[4:]

full_df['Title'].replace(to_replace=titles[4:],value='Other', inplace=True)

full_df['Title'].unique()
full_df[['MaidenName','NickName']] = full_df[['MaidenName','NickName']].notna().astype('int')

full_df.drop(['LastName','FirstName'], axis=1, inplace=True)

full_df.head()
def split_ticket(row):

    ticket_info = row['Ticket'].split()

    if len(ticket_info) == 1:

        try:

            int(ticket_info[0])

        except ValueError:

            return (ticket_info[0], 0)

        else:

            return (np.nan, int(ticket_info[0]))

    elif len(ticket_info) > 1:

        return (' '.join(ticket_info[:-1]), ticket_info[-1])

    

    
splitted_ticket = full_df.apply(split_ticket, axis=1, result_type='expand')

splitted_ticket.columns = ['TicketCode', 'TicketNumber']

full_df = full_df.join(splitted_ticket)

full_df['TicketNumber'] = full_df['TicketNumber'].astype('int')

full_df.drop('Ticket', axis=1, inplace=True)
full_df['TicketCode'].unique()

full_df.drop('TicketCode', axis=1, inplace=True)
boat_sections = full_df[full_df['Cabin'].notnull()]['Cabin'].apply(lambda x: x[0])

boat_sections.rename('BoatSection', inplace=True)

full_df = full_df.join(boat_sections, how='left')

full_df.drop('Cabin', axis=1, inplace=True)
corr_df = full_df.copy()
title_label = {'Mr': 1, 'Mrs': 2, 'Miss': 3, 'Master': 4, 'Other': 5} #Those labels are given arbitrary

corr_df['Title'].replace(title_label, inplace=True)
sections = {'None':0, 'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6,'G':7, 'T':8}

# 0 is imputed for passenger that have no cabin attributed in the dataset otherwise 1 is attributed to the section A in the considered first class, B is attributed 2 as considered a bit lower etc.

corr_df['BoatSection'].replace(sections, inplace=True)

corr_df['BoatSection'].fillna(value='0', inplace=True)

corr_df['BoatSection']= corr_df['BoatSection'].astype('int')
corr_df['IsFemale'] = (corr_df['Sex'] == 'female').astype('int') #IsFemale is used to be compatible with the is married field

corr_df.drop('Sex', axis=1, inplace=True)
corr_df['Embarked'].replace({'S':3, 'C':2, 'Q':1}, inplace=True)
plt.figure(figsize=(12,7))

sns.heatmap(corr_df.corr(), cmap='plasma', annot=True)
sns.set_style("whitegrid")

fig = plt.figure(figsize=(12,5))

sns.boxplot(x='Pclass', y='Fare', data = full_df)

plt.ylim(0,300)
# I use train dataset instead of full_df to prevent data leakage

median_fare_per_class = train[train['Fare'] != 0].groupby('Pclass')['Fare'].median() 

median_fare_per_class

def inpute_fare_on_class(row):

    if row['Fare']==0:

        return median_fare_per_class.loc[row['Pclass']]

    else:

        return row['Fare']
full_df['Fare'].fillna(value=0, inplace=True) # All missing values are converted to 0

full_df['Fare'] = full_df.apply(inpute_fare_on_class, axis=1) # Transform all the zeros coming from missing values but also fare of zeros which seems not absurd values. 
plt.figure(figsize=(12,10))

sns.boxplot(x='SibSp', y='Age', data=full_df, hue='Pclass')
plt.figure(figsize=(12,10))

sns.boxplot(x='Title', y='Age', data=full_df, hue='Pclass')
# Same as for Fare, I use only the value from train set to avoid leakage

median_value_for_age = pd.pivot_table(full_df[full_df['Survived'].notna()][['Age','Title', 'Pclass']], values='Age', columns=['Title'], index=['Pclass'], aggfunc=np.median)

median_value_for_age

median_value_for_age.loc[3,'Other'] = median_value_for_age.loc[[1,2],'Other'].mean()

median_value_for_age
def impute_age_on_class_title(row):

    if row['Age'] == 0:

        return median_value_for_age.loc[row['Pclass'],row['Title']]

    else:

        return row['Age']
full_df['Age'].fillna(value=0, inplace=True)

full_df['Age'] = full_df.apply(impute_age_on_class_title, axis=1)
full_df['Embarked'].fillna(value='S', inplace=True)

full_df['WithBoatSection'] = full_df['BoatSection'].notna().astype('int')
full_df.head(10)
full_df['IsFemale'] = (full_df['Sex'] == 'female').astype('int')

full_df.drop('Sex', axis=1, inplace=True)
# 0 is imputed for passenger that have no cabin attributed in the dataset otherwise 1 is attributed to the section A in the considered first class, B is attributed 2 as considered a bit lower etc.

full_df['BoatSection'].replace(sections, inplace=True)

full_df['BoatSection'].fillna(value='0', inplace=True)

full_df['BoatSection']= full_df['BoatSection'].astype('int')
full_df = pd.get_dummies(full_df)

full_df.drop(['Embarked_C', 'Title_Other'], axis=1, inplace=True)
full_df.head(10)
full_df.info()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

full_df[['Age', 'Fare', 'TicketNumber']] = scaler.fit_transform(full_df[['Age', 'Fare', 'TicketNumber']])
full_df
train = full_df[full_df['Survived'].notnull()].copy()

test = full_df[full_df['Survived'].isnull()].copy()

test.drop('Survived', axis=1, inplace=True)
from xgboost.sklearn import XGBClassifier

import xgboost as xgb

from sklearn import metrics

from sklearn.model_selection import train_test_split, cross_val_score
def modelfit(alg, dtrain, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    

    if useTrainCV:

        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(data=dtrain[predictors].values, label=dtrain[target].values)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics='error', early_stopping_rounds=early_stopping_rounds)

        alg.set_params(n_estimators=cvresult.shape[0])

    

    #Fit the algorithm on the data

    alg.fit(dtrain[predictors], dtrain[target], eval_metric='error')

        

    #Predict training set:

    dtrain_predictions = alg.predict(dtrain[predictors])

    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

        

    #Print model report:

    print("\nModel Report")

    print(f"Accuracy : {metrics.accuracy_score(dtrain[target].values, dtrain_predictions)}")

    print(f"AUC Score (Train): {metrics.roc_auc_score(dtrain[target], dtrain_predprob)}")

                    

    figure = plt.figure(figsize=(12,6))

    xgb.plot_importance(alg)

    plt.show()



# Initial model

model = XGBClassifier(silent=False,

                      min_child_weight=1,

                      scale_pos_weight=1,

                      learning_rate=0.3,  

                      colsample_bytree = 0.8,

                      subsample = 0.8,

                      objective='binary:logistic', 

                      n_estimators=1000, 

                      reg_alpha = 0.3,

                      max_depth=5, 

                      gamma=0)



X = train.drop(['Survived','PassengerId'], axis=1)

y = train['Survived']



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)



dtrain=pd.concat([X_train, y_train], axis=1)

predictors = X_train.columns.tolist()
modelfit(model, dtrain, predictors, 'Survived')
model.get_params()
from sklearn.model_selection import GridSearchCV



def grid_search(param_grid):

    gridsearch = GridSearchCV(model, param_grid, verbose=1, cv=5, n_jobs=4, scoring = ['accuracy', 'precision'], refit=False) 

    gridsearch.fit(X_train,y_train, eval_set=[(X_val, y_val)], eval_metric='error', verbose=False, early_stopping_rounds=50) # Use subset of Train as a validation set

    result = pd.DataFrame(gridsearch.cv_results_).sort_values(['rank_test_accuracy', 'rank_test_precision'])

    columns=[]

    for key in param_grid.keys():

        columns.append(f'param_{key}')



    columns = columns + ['mean_test_accuracy', 'std_test_accuracy','rank_test_accuracy','mean_test_precision','std_test_precision','rank_test_precision']

        

    return result[columns] # Does not output the best value but all the values sorted which help to get a feeling of the impact of the parameters
paramGrid1 = {"max_depth": range(3,12,2),

             "min_child_weight": range(1,8,2)}

result1 = grid_search(paramGrid1)

result1.head()



paramGrid2 = {"max_depth":[4,5,6],

             "min_child_weight":[4,5,6]}



result2 = grid_search(paramGrid2)

result2.head()
model.set_params(max_depth=4, min_child_weight=5)
paramGrid3 = {"gamma":[i/10 for i in range(11)],

              "subsample":[i/10.0 for i in range(11)],

              "colsample_bytree":[i/10.0 for i in range(11)]}



result3=grid_search(paramGrid3)

result3.head()
paramGrid4 = {"gamma":[i/100 for i in range(0, 80, 5)],

              "subsample":[i/100 for i in range(70,100, 5)],

              "colsample_bytree":[i/100 for i in range(25, 90, 5)]}



result4 = grid_search(paramGrid4)

result4.head()
model.set_params(gamma=0, subsample=0.8, colsample_bytree=0.75)
paramGrid5 = {'reg_alpha':[3*10**(i) for i in range(-5, 2)],

             'reg_lambda':[1*10**(i) for i in range(-5, 2)]}



result5 = grid_search(paramGrid5)

result5.head(10)
paramGrid7 = {'reg_alpha':[i/10 for i in range(1,11)],

             'reg_lambda':[i/10 for i in range(1,11)]}



result7 = grid_search(paramGrid7)



result7.head()
final_model = model.set_params(reg_alpha=0.4, reg_lambda=0.8)

validation = final_model.predict(X_val)
modelfit(final_model, dtrain, predictors, 'Survived', useTrainCV=False)
print(metrics.confusion_matrix(y_val, validation))

print('\n')

print(metrics.classification_report(y_val, validation))
final_model.fit(X,y)

test['Survived'] = final_model.predict(test.drop('PassengerId', axis=1))
submission_df = test[['PassengerId','Survived']].copy()

submission_df['Survived'] = submission_df['Survived'].astype('uint8')
submission_df.to_csv('submission_titanic.csv', index=False)
from sklearn.ensemble import RandomForestClassifier
rf_class = RandomForestClassifier()



param_grid = {

    'n_estimators':[10,30,50,100],

    'criterion':['gini','entropy'],

    'max_depth':[4,5,7,8,9],

    'min_samples_split':[2,4,6]

}



CV_rfc = GridSearchCV(estimator=rf_class, param_grid=param_grid, cv=5, verbose=3)

CV_rfc.fit(X,y)

final_model = CV_rfc.best_estimator_.fit(X,y)
test.drop('Survived', axis=1, inplace=True)
test['Survived'] = final_model.predict(test.drop('PassengerId', axis=1))
submission_rfc = test[['PassengerId','Survived']]

submission_rfc['Survived'] = submission_rfc['Survived'].astype('uint8')
submission_rfc.to_csv('submission_titanic_rfc.csv', index=False)