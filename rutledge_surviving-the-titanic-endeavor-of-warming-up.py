%matplotlib inline



import pandas as pd

import numpy as np

import scipy.stats as st

import seaborn as sns

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams

import re

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

from sklearn.model_selection import GridSearchCV, KFold

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn import cross_validation, metrics

import xgboost as xgb

from xgboost import XGBClassifier



## Import the training data and look at it

train_df = pd.read_csv('../input/train.csv', sep = ',', index_col = 0)

print(train_df.describe())

print('\nnumber of null values for each category\n', train_df.isnull().sum())

print(train_df.head())
sns.countplot(train_df['Pclass'])

print('Most passengers were third class')
sns.violinplot(x = 'Fare', y = 'Pclass', data = train_df, orient = 'h', width = 1)

print('Median fare for first class:', train_df['Fare'].loc[train_df['Pclass'] == 1].median())

print('Median fare for second class:', train_df['Fare'].loc[train_df['Pclass'] == 2].median())

print('Median fare for third class:', train_df['Fare'].loc[train_df['Pclass'] == 3].median())

print('\nThere is a clear association between Pclass and Fare, suggesting that this could be used to infer wealth.')
print('First class survival rate:', 100*136./(136+80))

print('Second class survival rate:', 100*87./(87+97))

print('Third class survival rate:', 100*119./(119+372))

print(train_df[['Pclass', 'Fare', 'Survived']].groupby(['Pclass', 'Survived']).count())

print('\nFirst class passengers clearly survived better than those in second or third class (63% vs 47% or 24%)')
## Make text lowercase

train_df['Name'] = train_df['Name'].apply(lambda x: x.lower())

## Extract title from the name

train_df['title'] = train_df['Name'].apply(lambda x: re.split('\.', re.split(', ', x)[1])[0])

## See how survival relates to title

train_df[['title', 'Survived', 'Name']].groupby(['title', 'Survived']).count()

print('I am defining titled as anything but Mr, Mrs, Ms, Miss, Master (designated boys back then), or Mlle')

## make a simple function to return 0 if the title passed is in a common list, otherwise return 1.

def title_finder(title):

    simple_titles = ['mr', 'mrs', 'ms', 'miss', 'mlle', 'master']

    if title in simple_titles:

        return 0

    else:

        return 1

train_df['has_title'] = train_df['title'].apply(title_finder)
print('Untitled survival rate:', 100*333./(333+534))

print('Titled class survival rate:', 100*9./(15+9))

print( train_df[['has_title', 'Name', 'Survived']].groupby(['has_title', 'Survived']).count())

print('\nTitled and untitled passengers fare similarly, but by eye it appears that there may be a significant gender gap in this metric')
print('Male untitled survival rate:', 100*104./(104+453))

print('Female untitled class survival rate:', 100*229./(229+81))

print('Male titled class survival rate:', 100*5./(15+5))

print('Female titled survival rate:', 100*4./(4+0))

print(train_df[['has_title', 'Sex', 'Name', 'Survived']].groupby(['has_title', 'Sex', 'Survived']).count())

print("Fisher's exact test p-value for males", st.fisher_exact([[104, 5],[453, 15]])[1])

print("Fisher's exact test p-value for females", st.fisher_exact([[229, 4],[81, 0]])[1])

## The low sample size for titled men and women make a [2x2]x[2x2] chi squared contingency table assessment unreliable

# print "Chi squared contingency table of survival by gender and title status:", st.chi2_contingency([[[229, 4],[81, 0]], [[104, 5],[453, 15]]])[1]

print('\nTitled passengers appear to do better by eye when gender is taken into account, but this is not supported by conservative statistics')
## Lowercase sex

train_df['Sex'] = train_df['Sex'].apply(lambda x: x.lower())

## Show gender differences here

print(train_df[['Sex', 'Name', 'Survived']].groupby(['Sex', 'Survived']).count())

print("There is an obvious difference in the survival rate by gender that is supported by statistics.")

print("Chi squared contingency table p-value with two-sided null hypothesis", st.chi2_contingency([[233, 109],[81, 468]])[1])
## Look at how age relates to survival

train_df['Age'].loc[train_df['Survived'] == 0].hist(bins = 100)

train_df['Age'].loc[train_df['Survived'] == 1].hist(bins = 100)
## Impute age

master_med_age = train_df['Age'].loc[train_df['title'] == 'Master'].median()

male_med_age = train_df['Age'].loc[train_df['Sex'] == 'male'].median()

female_med_age = train_df['Age'].loc[train_df['Sex'] == 'female'].median()



def median_age_assigner(row):

    ## Assigns null ages based on gender and title. Checks for the "Master" title

    ### to identify boys and returns the median age for boys for passengers with 

    ### this title but without an age. Passengers without this title are checked

    ### for gender and if their age is null and gender is female they are assigned

    ### the median age for all females. 

    row_age = row['Age']

    row_sex = row['Sex']

    row_title = row['title']

    if row.isnull().loc['Age']:

        if row_title == 'Master':

            return master_med_age

        elif row_sex == 'male':

            return male_med_age

        elif row_sex == 'female':

            return female_med_age

    else:

        return row_age

        

train_df['Age'] = train_df.apply(median_age_assigner, axis = 1)
## Look at how age relates to survival after imputation

train_df['Age'].loc[train_df['Survived'] == 0].hist(bins = 100)

train_df['Age'].loc[train_df['Survived'] == 1].hist(bins = 100)
train_df['Age'].loc[train_df['Survived'] == 0].loc[train_df['Age'] <= 21].hist(bins = 21)

train_df['Age'].loc[train_df['Survived'] == 1].loc[train_df['Age'] <= 21].hist(bins = 21)

print("It appears that survival drops off around age 18, so we'll use this as the age cutoff for children and make a new feature 'is_child'")
train_df['is_child'] = 0

train_df['is_child'].loc[train_df['Age'] < 18] = 1

print(train_df[['is_child', 'Name', 'Survived']].groupby(['is_child', 'Survived']).count())

print("Chi squared contingency table p-value with two-sided null hypothesis", st.chi2_contingency([[279, 63],[495, 54]])[1])

print("It appears that children/minors have a higher survival rate")
print(train_df[['SibSp', 'Name', 'Survived']].groupby(['SibSp', 'Survived']).count())

train_df['has_sibsp'] = 0

train_df['has_sibsp'].loc[train_df['SibSp'] > 0] = 1

print(train_df[['has_sibsp', 'Name', 'Survived']].groupby(['has_sibsp', 'Survived']).count())
print(train_df[['Parch', 'Name', 'Survived']].groupby(['Parch', 'Survived']).count())

train_df['has_parch'] = 0

train_df['has_parch'].loc[train_df['Parch'] > 0] = 1

print(train_df[['has_parch', 'Name', 'Survived']].groupby(['has_parch', 'Survived']).count())
train_df['family_size'] = train_df['SibSp'] + train_df['Parch'] + 1

print(train_df[['family_size', 'Name', 'Survived']].groupby(['family_size', 'Survived']).count())

print("Single people didn't do too well, and neither did members of large families (5 or more members). Small families (2-4 members), however, fared better than average.")
## Lowercase Ticket

train_df['Ticket'] = train_df['Ticket'].apply(lambda x: x.lower())

## Extract qualifier from the ticket

def get_ticket_qualifier(ticket):

    separated_ticket = re.split(' ', ticket)

    if len(separated_ticket) > 1:

        return separated_ticket[0]

    else:

        return 'none'

train_df['ticket_qualifier'] = train_df['Ticket'].apply(get_ticket_qualifier)

print(train_df['ticket_qualifier'].unique().tolist())
## Extract qualifier from the ticket

def get_ticket_qualifier(ticket):

    separated_ticket = re.split(' ', ticket)

    if len(separated_ticket) > 1:

        temp_qual = re.sub('[^A-Za-z0-9]+', '', separated_ticket[0])

        return temp_qual

    else:

        return 'none'

train_df['ticket_qualifier'] = train_df['Ticket'].apply(get_ticket_qualifier)

train_df['has_ticket_qualifier'] = 0

train_df['has_ticket_qualifier'].loc[train_df['ticket_qualifier'] != 'none'] = 1

print(train_df['ticket_qualifier'].unique().tolist())
print(train_df[['has_ticket_qualifier', 'Name', 'Survived']].groupby(['has_ticket_qualifier', 'Survived']).count())

print("Chi squared contingency table p-value with two-sided null hypothesis", st.chi2_contingency([[255, 87],[410, 139]])[1])

print(train_df[['ticket_qualifier', 'Name', 'Survived']].groupby(['ticket_qualifier', 'Survived']).count())

def does_ticket_start_with_p(ticket_text):

    if str(ticket_text)[0].lower() == 'p':

        return 1

    else:

        return 0

train_df['ticket_starts_with_p'] = train_df['Ticket'].apply(does_ticket_start_with_p)

print(train_df[['ticket_starts_with_p', 'Name', 'Survived']].groupby(['ticket_starts_with_p', 'Survived']).count())

print("Chi squared contingency table p-value with two-sided null hypothesis", st.chi2_contingency([[300, 42],[526, 23]])[1])
sns.violinplot(x = 'Fare', y = 'Survived', data = train_df, orient = 'h')

print(st.ttest_ind(train_df['Fare'].loc[train_df['Survived'] == 0], train_df['Fare'].loc[train_df['Survived'] == 1]))

print("People who paid higher fares were more likely to survive.")
## Lowercase cabin

train_df['Cabin'] = train_df['Cabin'].dropna().apply(lambda x: x.lower())

print(train_df['Cabin'].unique().tolist())

print("\nnumber of nulls in cabin", train_df['Cabin'].isnull().sum())

plt.figure(figsize = [9,4])

sns.countplot(train_df['Cabin'].dropna())
## Add had_cabin_assigment to train_df

train_df['had_cabin_assignment'] = 0

train_df['had_cabin_assignment'].loc[train_df['Cabin'].isnull() != True] = 1

## Fill NaNs with None

train_df['Cabin'].fillna('None', inplace = True)

print(train_df[['had_cabin_assignment', 'Survived', 'Name']].groupby(['had_cabin_assignment', 'Survived']).count())
sns.violinplot(x = 'Fare', y = 'had_cabin_assignment', data = train_df, orient = 'h')
print(train_df['Embarked'].unique().tolist())

print("\nnumber of nulls in embarked", train_df['Embarked'].isnull().sum())

sns.countplot(train_df['Embarked'].dropna())
## Impute nans to S

train_df['Embarked'].fillna('S', inplace = True)

## Make Embarked lowercase

train_df['Embarked'] = train_df['Embarked'].apply(lambda x: x.lower())

print(train_df[['Embarked', 'Name', 'Survived']].groupby(['Embarked', 'Survived']).count())

print("Chi squared contingency table p-value with two-sided null hypothesis", st.chi2_contingency([[93, 30, 217],[75, 47, 427]])[1])
sns.violinplot(x = 'Fare', y = 'Embarked', data = train_df, orient = 'h')

print(st.ttest_ind(train_df['Fare'].loc[train_df['Embarked'] == 'C'], train_df['Fare'].loc[train_df['Embarked'] != 'C']))

print("Passengers who embarked from 'C' clearly paid more for their trip, which could be an indicator of wealth.")
def data_processor(df):

    ## Set embarked to categorical integers

    df['Embarked'].fillna('S', inplace = True)

    df['emb_s'] = 0

    df['emb_s'].loc[df['Embarked'] == 'S'] = 1

    df['emb_c'] = 0

    df['emb_c'].loc[df['Embarked'] == 'C'] = 1

    df['emb_q'] = 0

    df['emb_q'].loc[df['Embarked'] == 'Q'] = 1

    ## Set new feature "had_cabin"

    df['had_cabin'] = 0

    df['had_cabin'].loc[df['Cabin'].isnull() == False] = 1

    ## Get title

    df['Name'] = df['Name'].apply(lambda x: x.lower())

    ### Extract title from the name

    df['title'] = df['Name'].apply(lambda x: re.split('\.', re.split(', ', x)[1])[0])

    ### make a simple function to return 0 if the title passed is in a common list, otherwise return 1.

    def title_finder(title):

        simple_titles = ['mr', 'mrs', 'ms', 'miss', 'mlle', 'master']

        if title in simple_titles:

            return 0

        else:

            return 1

    df['had_title'] = df['title'].apply(title_finder)

    ## Impute age

    master_med_age = df['Age'].loc[df['title'] == 'Master'].median()

    male_med_age = df['Age'].loc[df['Sex'] == 'male'].median()

    female_med_age = df['Age'].loc[df['Sex'] == 'female'].median()



    def median_age_assigner(row):

        ## Assigns null ages based on gender and title. Checks for the "Master" title

        ### to identify boys and returns the median age for boys for passengers with 

        ### this title but without an age. Passengers without this title are checked

        ### for gender and if their age is null and gender is female they are assigned

        ### the median age for all females. 

        row_age = row['Age']

        row_sex = row['Sex']

        row_title = row['title']

        if row.isnull().loc['Age']:

            if row_title == 'Master':

                return master_med_age

            elif row_sex == 'male':

                return male_med_age

            elif row_sex == 'female':

                return female_med_age

        else:

            return row_age



    df['Age'] = df.apply(median_age_assigner, axis = 1).astype(int)

#     df['Age'].fillna(df['Age'].median(), inplace = True)

    ## Set new feature "child"

    df['child'] = 0

    df['child'].loc[df['Age'] < 18] = 1

    ## Set sex to categorical integers

    df['Sex'].loc[df['Sex'] != 'male'] = 0

    df['Sex'].loc[df['Sex'] == 'male'] = 1

    df['Sex'] = df['Sex'].astype(int)

    ## Family size

    df['family_size'] = df['SibSp'] + df['Parch'] + 1

    ## Ticket qualifier

    df['Ticket'] = df['Ticket'].apply(lambda x: x.lower())

    ## Extract qualifier from the ticket

    def get_ticket_qualifier(ticket):

        separated_ticket = re.split(' ', ticket)

        if len(separated_ticket) > 1:

            temp_qual = re.sub('[^A-Za-z0-9]+', '', separated_ticket[0])

            return temp_qual

        else:

            return 'none'

    df['ticket_qualifier'] = df['Ticket'].apply(get_ticket_qualifier)

    df['had_ticket_qualifier'] = 0

    df['had_ticket_qualifier'].loc[df['ticket_qualifier'] != 'none'] = 1

    def does_ticket_start_with_p(ticket_text):

        if str(ticket_text)[0].lower() == 'p':

            return 1

        else:

            return 0

    df['ticket_starts_with_p'] = df['Ticket'].apply(does_ticket_start_with_p)

    ## Fare

    pc1_fare_med = df['Fare'].loc[df['Pclass'] == 1].median()

    pc2_fare_med = df['Fare'].loc[df['Pclass'] == 2].median()

    pc3_fare_med = df['Fare'].loc[df['Pclass'] == 3].median()

    def fare_imputer(row):

        if row.isnull().sum() > 0:

            pc = row['Pclass']

            if pc == 1:

                return pc1_fare_med

            elif pc == 2:

                return pc2_fare_med

            else:

                return pc3_fare_med

        else:

            return row['Fare']

    df['Fare'] = df.apply(fare_imputer, axis = 1)

    ## Set fare above 75 to 75

#     df['Fare']

#     df['Fare'].loc[df['Fare'] > 75] = 75

#     df['Fare'] = df['Fare'].apply(round).astype(int)

    ## Pclass

    df['first_class'] = 0

    df['first_class'].loc[df['Pclass'] == 1] = 1

    df['second_class'] = 0

    df['second_class'].loc[df['Pclass'] == 2] = 1

    df['third_class'] = 0

    df['third_class'].loc[df['Pclass'] == 3] = 1

    return df
train_proc = data_processor(train_df)
train_proc.head()
train_x = train_proc.drop(['Pclass', 'Name', 'title', 'Ticket', 'ticket_qualifier', 'Cabin', 'Embarked', 'Survived'], axis = 1)

train_y = train_proc['Survived']
train_x.head()
lr = LogisticRegression(random_state = 0)

kf = KFold(n_splits=3, random_state = 0)

parameters = {'C': [i/10. for i in range(1, 50)]}



lr_gscv = GridSearchCV(lr, parameters, cv = kf, verbose = 1)

lr_gscv.fit(train_x, train_y)



print('Using all the features with a logistic regression scored', lr_gscv.score(train_x, train_y))

print('')

print(sns.barplot(y = train_x.columns, x = np.abs(lr_gscv.best_estimator_.coef_[0]), orient = 'h'))

print('')

print( "So it looks like Sex, has_sibsp, Pclass, Sibsp, and is_child are the top 5 features that the logistic regression likes.")

print('')

print( "It also appears that the features that are binary or only contain a small amount of categories are more predictive, which makes sense given the way the classifier works.")
kf = KFold(n_splits=3, random_state = 0)



parameters = {'C': [0.7], 

              'kernel': ['linear']}

svc = SVC(decision_function_shape = 'ovr', random_state = 0)

svc_gscv = GridSearchCV(svc, parameters, cv = kf, verbose = 2)

svc_gscv.fit(train_x, train_y)

print(svc_gscv.best_params_)

print('')

print('Using all the features with a svc scored', np.mean(svc_gscv.best_score_))

print(sns.barplot(y = train_x.columns, x = np.abs(svc_gscv.best_estimator_.coef_[0]), orient = 'h'))
from sklearn.naive_bayes import GaussianNB

kf = KFold(n_splits=3, random_state = 0)

gnb = GaussianNB()

gnb.fit(train_x, train_y)

print('Using all the features with a GNB classifier scored', gnb.score(train_x, train_y))
parameters = {

    'n_estimators': [100, 125, 150], 

    'max_features': [5, 10, 15],

    'max_depth': (2, 5, 10, 15),

    'min_samples_split': [2, 3, 5, 10], 

    'min_samples_leaf': [1, 2, 3, 5, 10]

}



kf = KFold(n_splits = 3, random_state = 0)



rfc = RandomForestClassifier(random_state = 0)

rfc_gscv = GridSearchCV(rfc, parameters, cv = kf, verbose = 1)

rfc_gscv.fit(train_x, train_y)



print(rfc_gscv.best_params_)

print("")

print('Using all the features with a random forest classifier scored', np.mean(rfc_gscv.best_score_))

print(sns.barplot(y = train_x.columns, x = np.abs(rfc_gscv.best_estimator_.feature_importances_), orient = 'h'))
## Set the grid search parameters for the xgb classifier to pass to GridSearchCV



parameters = {

    'n_estimators': [110],

    'max_depth':range(3, 10, 2),

    'min_child_weight':range(1, 6, 2),

    #'gamma': [i/10.0 for i in range(0, 6)],

    #'subsample': [i/10.0 for i in range(5, 11)],

    #'colsample_bytree': [i/10.0 for i in range(5, 11)],

    #'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 100],

    #'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 100],

    #'learning_rate': [i/100. for i in range(1, 15)]

}



training_data = train_x.join(train_y)



predictors = training_data.drop('Survived', axis = 1).columns

outcome = 'Survived'



## Instantiate the xgb classifier

xgbc = XGBClassifier(learning_rate =0.1, n_estimators=110, max_depth=5, min_child_weight=1, 

                     gamma=0, subsample=0.8, colsample_bytree=0.8, 

                     objective= 'binary:logistic', 

                     nthread=-1, seed=0, silent = False)

## Set up the k folds function to pass to GridSearchCV

kf = KFold(n_splits = 3, random_state = 0)

## Instantiate the grid search across the parameters and with cross validation on 3 folds

xgbc_gscv = GridSearchCV(xgbc, parameters, cv = kf, verbose = 2)

## Do the grid search and cv

xgbc_gscv.fit(training_data[predictors], training_data[outcome])



print(xgbc_gscv.best_params_)

print(xgbc_gscv.best_score_)

feat_imp = pd.Series(xgbc_gscv.best_estimator_.booster().get_fscore()).sort_values(ascending=False)

print(feat_imp.plot(kind='bar', title='Feature Importances'))
## Set the grid search parameters for the xgb classifier to pass to GridSearchCV



parameters = {

    'n_estimators': [110],

    'max_depth': [5],

    'min_child_weight': [5],

    'gamma': [i/10.0 for i in range(0, 6)],

    #'subsample': [i/10.0 for i in range(5, 11)],

    #'colsample_bytree': [i/10.0 for i in range(5, 11)],

    #'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 100],

    #'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 100],

    #'learning_rate': [i/100. for i in range(1, 15)]

}



training_data = train_x.join(train_y)



predictors = training_data.drop('Survived', axis = 1).columns

outcome = 'Survived'



## Instantiate the xgb classifier

xgbc = XGBClassifier(learning_rate =0.1, n_estimators=110, max_depth=5, min_child_weight=1, 

                     gamma=0, subsample=0.8, colsample_bytree=0.8, 

                     objective= 'binary:logistic', 

                     nthread=-1, seed=0, silent = False)

## Set up the k folds function to pass to GridSearchCV

kf = KFold(n_splits = 3, random_state = 0)

## Instantiate the grid search across the parameters and with cross validation on 3 folds

xgbc_gscv = GridSearchCV(xgbc, parameters, cv = kf, verbose = 2)

## Do the grid search and cv

xgbc_gscv.fit(training_data[predictors], training_data[outcome])



print(xgbc_gscv.best_params_)

print(xgbc_gscv.best_score_)

feat_imp = pd.Series(xgbc_gscv.best_estimator_.booster().get_fscore()).sort_values(ascending=False)

print(feat_imp.plot(kind='bar', title='Feature Importances'))
## Set the grid search parameters for the xgb classifier to pass to GridSearchCV



parameters = {

    'n_estimators': [110],

    'max_depth': [5],

    'min_child_weight': [5],

    'gamma': [0],

    'subsample': [i/10.0 for i in range(5, 11)],

    'colsample_bytree': [i/10.0 for i in range(5, 11)],

    #'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 100],

    #'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 100],

    #'learning_rate': [i/100. for i in range(1, 15)]

}



training_data = train_x.drop(['Name', 'Ticket'], axis = 1).join(train_y)



predictors = training_data.drop('Survived', axis = 1).columns

outcome = 'Survived'



## Instantiate the xgb classifier

xgbc = XGBClassifier(learning_rate =0.1, n_estimators=110, max_depth=5, min_child_weight=1, 

                     gamma=0, subsample=0.8, colsample_bytree=0.8, 

                     objective= 'binary:logistic', 

                     nthread=-1, seed=0, silent = False)

## Set up the k folds function to pass to GridSearchCV

kf = KFold(n_splits = 3, random_state = 0)

## Instantiate the grid search across the parameters and with cross validation on 3 folds

xgbc_gscv = GridSearchCV(xgbc, parameters, cv = kf, verbose = 2)

## Do the grid search and cv

xgbc_gscv.fit(training_data[predictors], training_data[outcome])



print(xgbc_gscv.best_params_)

print(xgbc_gscv.best_score_)

feat_imp = pd.Series(xgbc_gscv.best_estimator_.booster().get_fscore()).sort_values(ascending=False)

print(feat_imp.plot(kind='bar', title='Feature Importances'))
## Set the grid search parameters for the xgb classifier to pass to GridSearchCV



parameters = {

    'n_estimators': [110],

    'max_depth': [5],

    'min_child_weight': [5],

    'gamma': [0],

    'subsample': [1],

    'colsample_bytree': [0.7],

    'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 100],

    'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 100],

    #'learning_rate': [i/100. for i in range(1, 15)]

}



training_data = train_x.drop(['Name', 'Ticket'], axis = 1).join(train_y)



predictors = training_data.drop('Survived', axis = 1).columns

outcome = 'Survived'



## Instantiate the xgb classifier

xgbc = XGBClassifier(learning_rate =0.1, n_estimators=110, max_depth=5, min_child_weight=1, 

                     gamma=0, subsample=0.8, colsample_bytree=0.8, 

                     objective= 'binary:logistic', 

                     nthread=-1, seed=0, silent = False)

## Set up the k folds function to pass to GridSearchCV

kf = KFold(n_splits = 3, random_state = 0)

## Instantiate the grid search across the parameters and with cross validation on 3 folds

xgbc_gscv = GridSearchCV(xgbc, parameters, cv = kf, verbose = 2)

## Do the grid search and cv

xgbc_gscv.fit(training_data[predictors], training_data[outcome])



print(xgbc_gscv.best_params_)

print(xgbc_gscv.best_score_)

feat_imp = pd.Series(xgbc_gscv.best_estimator_.booster().get_fscore()).sort_values(ascending=False)

print(feat_imp.plot(kind='bar', title='Feature Importances'))
## Set the grid search parameters for the xgb classifier to pass to GridSearchCV



parameters = {

    'n_estimators': [110],

    'max_depth': [5],

    'min_child_weight': [5],

    'gamma': [0],

    'subsample': [1],

    'colsample_bytree': [0.7],

    'reg_alpha': [0.1],

    'reg_lambda': [0],

    'learning_rate': [i/100. for i in range(1, 15)]

}



training_data = train_x.drop(['Name', 'Ticket'], axis = 1).join(train_y)



predictors = training_data.drop('Survived', axis = 1).columns

outcome = 'Survived'



## Instantiate the xgb classifier

xgbc = XGBClassifier(learning_rate =0.1, n_estimators=110, max_depth=5, min_child_weight=1, 

                     gamma=0, subsample=0.8, colsample_bytree=0.8, 

                     objective= 'binary:logistic', 

                     nthread=-1, seed=0, silent = False)

## Set up the k folds function to pass to GridSearchCV

kf = KFold(n_splits = 3, random_state = 0)

## Instantiate the grid search across the parameters and with cross validation on 3 folds

xgbc_gscv = GridSearchCV(xgbc, parameters, cv = kf, verbose = 2)

## Do the grid search and cv

xgbc_gscv.fit(training_data[predictors], training_data[outcome])



print(xgbc_gscv.best_params_)

print(xgbc_gscv.best_score_)

feat_imp = pd.Series(xgbc_gscv.best_estimator_.booster().get_fscore()).sort_values(ascending=False)

print(feat_imp.plot(kind='bar', title='Feature Importances'))
## Set the grid search parameters for the xgb classifier to pass to GridSearchCV



parameters = {

    'n_estimators': [110],

    'max_depth': [5],

    'min_child_weight': [5],

    'gamma': [0],

    'subsample': [1],

    'colsample_bytree': [0.7],

    'reg_alpha': [0.1],

    'reg_lambda': [0],

    'learning_rate': [0.12]

}



training_data = train_x.join(train_y)



predictors = training_data.drop('Survived', axis = 1).columns

outcome = 'Survived'



## Instantiate the xgb classifier

xgbc = XGBClassifier(learning_rate =0.1, n_estimators=110, max_depth=5, min_child_weight=1, 

                     gamma=0, subsample=0.8, colsample_bytree=0.8, 

                     objective= 'binary:logistic', 

                     nthread=-1, seed=0, silent = False)

## Set up the k folds function to pass to GridSearchCV

kf = KFold(n_splits = 3, random_state = 0)

## Instantiate the grid search across the parameters and with cross validation on 3 folds

xgbc_gscv = GridSearchCV(xgbc, parameters, cv = kf, verbose = 2)

## Do the grid search and cv

xgbc_gscv.fit(training_data[predictors], training_data[outcome])



print(xgbc_gscv.best_params_)

print(xgbc_gscv.best_score_)

feat_imp = pd.Series(xgbc_gscv.best_estimator_.booster().get_fscore()).sort_values(ascending=False)

print(feat_imp.plot(kind='bar', title='Feature Importances'))
log_reg1 = LogisticRegressionCV(cv = 3, random_state = 0)

log_reg2 = LogisticRegressionCV(cv = 3, random_state = 0)



log_reg1.fit(train_x.drop(['Name', 'Ticket'], axis = 1), train_y)

log_reg2.fit(train_x[['Sex', 'has_sibsp', 'Pclass', 'SibSp', 'is_child']], train_y)



print('Using all the features with a logistic regression scored', log_reg1.score(train_x.drop(['Name', 'Ticket'], axis = 1), train_y))

print('Using the top 5 features with a logistic regression scored', log_reg2.score(train_x[['Sex', 'has_sibsp', 'Pclass', 'SibSp', 'is_child']], train_y))
## Set the grid search parameters for the xgb classifier to pass to GridSearchCV

parameters = {

    'n_estimators': [110],

    'max_depth': [5],

    'min_child_weight': [5],

    'gamma': [0],

    'subsample': [1],

    'colsample_bytree': [0.7],

    'reg_alpha': [0.1],

    'reg_lambda': [0],

    'learning_rate': [0.12]

}



## Instantiate the xgb classifier

xgbc = XGBClassifier(learning_rate =0.1, n_estimators=110, max_depth=5, min_child_weight=1, 

                     gamma=0, subsample=0.8, colsample_bytree=0.8, 

                     objective= 'binary:logistic', 

                     nthread=-1, seed=0, silent = False)

## Set up the k folds function to pass to GridSearchCV

kf = KFold(n_splits = 3, random_state = 0)

## Instantiate the grid search across the parameters and with cross validation on 3 folds

xgbc_gscv1 = GridSearchCV(xgbc, parameters, cv = kf, verbose = 1)

xgbc_gscv2 = GridSearchCV(xgbc, parameters, cv = kf, verbose = 1)

## Do the grid search and cv

xgbc_gscv1.fit(train_x.drop(['Name', 'Ticket'], axis = 1), train_y)

xgbc_gscv2.fit(train_x[['Fare', 'Age', 'Cabin', 'title', 'Embarked']], train_y)



print('XGB classifier with all features scored', xgbc_gscv1.best_score_)

print('XGB classifier with top 5 features scored', xgbc_gscv2.best_score_)
from sklearn.model_selection import train_test_split



## First, I want to make subsets of the training data to use as input for prediction models

train_ins = []

test_ins = []

train_outs = []

test_outs = []



for tts_ind in range(0,10):

    train_in, test_in, train_out, test_out = train_test_split(train_x,

                                                              train_y, 

                                                              test_size = 0.33,

                                                              random_state = tts_ind)

    train_ins.append(train_in)

    test_ins.append(test_in)

    train_outs.append(train_out)

    test_outs.append(test_out)

    

#print("First 5 ids of split 1: train -", train_ins[0].head().index.tolist(), "; test -", train_outs[0].head().index.tolist())

#print("First 5 ids of split 2: train -", train_ins[1].head().index.tolist(), "; test -", train_outs[1].head().index.tolist())
## Next, I need to make predictions from multiple models using each subset of data



### make correlation vectors

lr_svc_corrs = []

lr_xgb_corrs = []

svc_xgb_corrs = []



### set up the models

lr_pred_corr = LogisticRegression(C = 0.7, random_state = 0)

svc_pred_corr = SVC(C = 0.7, kernel = 'linear', decision_function_shape = 'ovr', 

                    random_state = 0)

xgb_pred_corr = XGBClassifier(learning_rate =0.12, n_estimators=110, max_depth=5, 

                              min_child_weight=5, gamma=0, subsample=1,

                              reg_alpha = 0.1, reg_lambda = 0,

                              colsample_bytree=0.7, objective= 'binary:logistic',

                              seed=0, silent = False)



for pred_ind in range(0,10):

    ## Predict the logistic regression outcomes of the test set

    lr_pred_corr.fit(train_ins[pred_ind], train_outs[pred_ind])

    lr_preds_temp = lr_pred_corr.predict(test_ins[pred_ind])

    ## Predict the svc classifier outcomes of the test set

    svc_pred_corr.fit(train_ins[pred_ind], train_outs[pred_ind])

    svc_preds_temp = svc_pred_corr.predict(test_ins[pred_ind])

    ## Predict the xgb classifier outcomes of the test set

    xgb_pred_corr.fit(train_ins[pred_ind], train_outs[pred_ind])

    xgb_preds_temp = xgb_pred_corr.predict(test_ins[pred_ind])

    ## Append correlations to correlation vectors

    lr_svc_corrs.append(st.pearsonr(lr_preds_temp, svc_preds_temp)[0])

    lr_xgb_corrs.append(st.pearsonr(lr_preds_temp, xgb_preds_temp)[0])

    svc_xgb_corrs.append(st.pearsonr(svc_preds_temp, xgb_preds_temp)[0])

    

print('lr vs svc:', np.mean(lr_svc_corrs), lr_svc_corrs)

print('lr vs xgb:', np.mean(lr_xgb_corrs), lr_xgb_corrs)

print('svc vs xgb:', np.mean(svc_xgb_corrs), svc_xgb_corrs)
## Compare predictions

lr_stack = LogisticRegression(C = 0.7, random_state = 0)

svc_stack = SVC(C = 0.7, decision_function_shape = 'ovr', random_state = 0)

rfc_stack = RandomForestClassifier(random_state = 0, max_features = 15, 

                                   min_samples_split = 10, max_depth = 15, 

                                   n_estimators = 150, min_samples_leaf = 2)

xgb_stack = XGBClassifier(learning_rate =0.12, n_estimators=110, max_depth=5, 

                           min_child_weight=5, gamma=0, subsample=1,

                           reg_alpha = 0.1, reg_lambda = 0,

                           colsample_bytree=0.7, objective= 'binary:logistic',

                           seed=0, silent = False)

from sklearn.naive_bayes import GaussianNB

gnb_stack = GaussianNB()



def make_preds(train_in, train_out, test_in):

    lr_stack.fit(train_in, train_out)

    gnb_stack.fit(train_in, train_out)

    svc_stack.fit(train_in, train_out)

    rfc_stack.fit(train_in, train_out)

    xgb_stack.fit(train_in, train_out)

    lr_stack_preds = lr_stack.predict(test_in)

    svc_stack_preds = svc_stack.predict(test_in)

    gnb_stack_preds = gnb_stack.predict(test_in)

    rfc_stack_preds = rfc_stack.predict(test_in)

    xgb_stack_preds = xgb_stack.predict(test_in)

    stack_preds = pd.DataFrame({'lr_preds': lr_stack_preds,

                                #'gnb_preds': gnb_stack_preds,

                                'svc_preds': svc_stack_preds, 

                                'rfc_preds': rfc_stack_preds,

                                'xgb_preds':xgb_stack_preds})

    return stack_preds



preds_0 = make_preds(train_ins[0], train_outs[0], test_ins[0])

stacking_classifier = LogisticRegression(random_state = 0)

stacking_classifier.fit(preds_0, test_outs[0])

xgb_stack.fit(train_ins[0], train_outs[0])

for pred_ind in range(1,10):

    preds_1 = make_preds(train_ins[pred_ind], train_outs[pred_ind], test_ins[pred_ind])

    print('Score', 

          pred_ind, 

          'stack -', 

          round(stacking_classifier.score(preds_1, 

                                          test_outs[pred_ind]), 3), 

          '; XGBC -', 

          round(xgb_stack.score(test_ins[pred_ind],

                                test_outs[pred_ind]), 3),

          '; RFC -', 

          round(rfc_stack.score(test_ins[pred_ind], 

                                test_outs[pred_ind]), 3), 

          '; SVC -', 

          round(svc_stack.score(test_ins[pred_ind], 

                                test_outs[pred_ind]), 3),

          #'; GNB -', 

          #round(gnb_stack.score(test_ins[pred_ind], 

          #                      test_outs[pred_ind]), 3),

          '; LR -', 

          round(lr_stack.score(test_ins[pred_ind], 

                               test_outs[pred_ind]), 3))

    

print(sns.barplot(y = preds_0.columns, x = np.abs(stacking_classifier.coef_[0]), orient = 'h'))