#import every library needed for completing task



# utils library 

import operator

import gc



# data wrangling

import pandas as pd

import numpy as np

from scipy import stats



# data visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import shuffle

from xgboost.sklearn import XGBClassifier

from sklearn import cross_validation, metrics
# make custom functions which is needed 



def plot_correlation_map(df):

    """Plot heatmap to expresss dataframe columns' correlationship"""

    corr = df.corr()

    _ , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

    )

    

def modelfit(alg, dtrain, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    """Return xgb model which is applied with the best parameters by cross-validation"""

    if useTrainCV:

        #need to pass the params which is verified by gridsearch

        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,

            metrics='auc', early_stopping_rounds=early_stopping_rounds)

        alg.set_params(n_estimators=cvresult.shape[0])

    

    #Fit the algorithm on the data

    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')

        

    #Predict training set:

    dtrain_predictions = alg.predict(dtrain[predictors])

    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

        

    #Print model report:

    print("\nModel Report")

    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))

    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

                    

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)

    feat_imp.plot(kind='bar', title='Feature Importances')

    plt.ylabel('Feature Importance Score')

    return alg
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

submission_exp = pd.read_csv('../input/gendermodel.csv')



combine = [train, test]



train.shape, test.shape, submission_exp.shape
train.describe()



# let's find out whether age, fare colums has correlation with the result

# Create the new features using 'SibSp' and 'Parch' to like 'company_num
train.describe(include=['O'])



# Think that it will be the issue how to convert Name, Ticket, Cabin columns values to numeric values later
train.Name[:5]



# Let's make new columns using name columns to check out whether the passengers is married or not and is male or female

# Mr. / Miss. / ....
train.Ticket[:5]



# Can't find out any meaning with the ticket number, maybe I will delete this later
train.Cabin[:5]



# I want to split the alphabet and the number. However, I am afraid of the fact that there are so many Nan values.

# Should come up with how to fill out the Nan values
plot_correlation_map(train)



# Found that Fare is correlated with survival a little bit

# Found no correlation between Age and Survivied (Let's just keep it first, and let's expriment what brings the better results)
train_nan_cnt = train.isnull().sum(axis=0).reset_index()

train_nan_cnt.columns = ['col_name', 'nan_count']

train_nan_cnt['nan_ratio'] = train_nan_cnt['nan_count'] / train.shape[0]

train_nan_cnt = train_nan_cnt.loc[train_nan_cnt.nan_ratio > 0]

train_nan_cnt = train_nan_cnt.sort_values('nan_ratio', ascending=False)



test_nan_cnt = test.isnull().sum(axis=0).reset_index()

test_nan_cnt.columns = ['col_name', 'nan_count']

test_nan_cnt['nan_ratio'] = test_nan_cnt['nan_count'] / test.shape[0]

test_nan_cnt = test_nan_cnt.loc[test_nan_cnt.nan_ratio > 0]

test_nan_cnt = test_nan_cnt.sort_values('nan_ratio', ascending=False)



train_nan_cnt
print("train: ", train_nan_cnt.col_name.values, "test: ", test_nan_cnt.col_name.values)
fig, ax = plt.subplots(ncols=2, figsize=(30, 20))



ax[0].set_title('train dataset')

ax[1].set_title('test dataset')



sns.barplot(train_nan_cnt.nan_ratio, train_nan_cnt.col_name, ax=ax[0])

sns.barplot(test_nan_cnt.nan_ratio, test_nan_cnt.col_name, ax=ax[1])



# fill out the age nan value with median of age

# fill out the embarked nan value with mode of embarked

# fill out the Fare nan value with median of Fare

# drop the cabin columns cause we don't have enough data for it and hard to find out what meaning the column has
test.describe(include=['O'])
# drop the column 'ticket' cause I didn't find out any meaing of it



for dataset in combine:

    dataset.drop('Ticket', axis=1, inplace=True)

    

train.head()
# create company_num columns by adding SibSp and Parch



for dataset in combine:

    dataset['company_num'] = dataset['SibSp'] + dataset['Parch']

    dataset.drop(['SibSp', 'Parch'], axis=1, inplace=True)

    

train.head()
# create new columns person_type with Name



for dataset in combine:

    dataset['person_type'] = dataset['Name'].apply(lambda x: x.split()[1])

    dataset.drop('Name', axis=1, inplace=True)



train.head()
# fill out the abnormal data by mode of this columns' values



train.person_type.value_counts()
dataset.loc[dataset.Cabin=='B57 B59 B63 B66']
# drop the columns 'cabin'



for dataset in combine:

    dataset.drop('Cabin', axis=1, inplace=True)

    

train.head()
# find out the most frequent value or the median value to fill out the Nan values



pre_final = pd.concat([train, test])



freq_age = pre_final.Age.median()



#only for train dataset

freq_embarked = pre_final.Embarked.mode()[0]



#only for test dataset

freq_fare = pre_final.Fare.median()
# fill out age values which is nan value right now



for dataset in combine:

    dataset['Age'].fillna(freq_age, inplace=True)

    

# fill out embarked and fare values which is nan in each dataset

train['Embarked'].fillna(freq_embarked, inplace=True)

test['Fare'].fillna(freq_fare, inplace=True)
# done with filling the nan values. Good!



print(train.isnull().sum(axis=0))

test.isnull().sum(axis=0)
# make the strange person type values to be right using freq_person_type



freq_person_type = pre_final.person_type.mode()[0]



train_strange_type = [

    'Planke,', 'Don.', 'Rev.',  'Billiard,', 'der', 'Walle,', 'Pelsmaeker,', 'Mulder,',

    'y', 'Steen,', 'Carlo,', 'Mme.', 'Impe,','Major.', 'Gordon,', 'Messemaeker,', 'Mlle.',

    'Col.', 'Capt.', 'Velde,', 'the', 'Shawah,', 'Jonkheer.', 'Melkebeke,', 'Cruyssen,'

]



test_strange_type = [

    'Carlo,', 'Khalil,', 'Master.', 'y', 

     'Palmquist,', 'Col.', 'Planke,', 'Rev.', 'Billiard,',

     'Messemaeker,', 'Brito,'

]



train['person_type'] = train['person_type'].apply(lambda x: x if x not in train_strange_type else freq_person_type)

test['person_type'] = test['person_type'].apply(lambda x: x if x not in test_strange_type else freq_person_type)
train.head()
# convert columns Sex, Embarked, person_type to oridinal values

# OK! done with preprocessing dataset for modeling





for dataset in combine:

    dataset['Sex'] = dataset['Sex'].astype('category').cat.codes

    dataset['Embarked'] = dataset['Embarked'].astype('category').cat.codes

    dataset['person_type'] = dataset['person_type'].astype('category').cat.codes
train.head()
test.head()
XGBClassifier()
clf = XGBClassifier()

predictors = [x for x in train.columns if x not in ['PassengerId', 'Survived']]



best_model = modelfit(clf, train, predictors, 'Survived')



# The column Age has high correlation with the result. No need to drop the column
# Let's predict with the model we trained now

test = test.set_index('PassengerId')



y_pred = best_model.predict(test)
# fit our result into the suggested forms for submission



xgb_submission = submission_exp

xgb_submission['Survived'] = y_pred



xgb_submission = xgb_submission.set_index('PassengerId')

xgb_submission.head()
# export to csv files 

xgb_submission.to_csv('xgb_result.csv')