# Imports

import numpy as np

import pandas as pd

import pandas_profiling as pp

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from matplotlib import pyplot as plt

import seaborn as sns

from xgboost import XGBClassifier

from sklearn.metrics import mean_squared_error as mse



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score



from sklearn.experimental import enable_iterative_imputer # For multivariate imputation

from fancyimpute import IterativeImputer

from sklearn.impute import IterativeImputer



# I/O

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Loading data

train_original = pd.read_csv('/kaggle/input/titanic/train.csv')

test_original = pd.read_csv('/kaggle/input/titanic/test.csv')



# Making copies to clean so I keep the originals seperate

train = train_original.copy()

test = test_original.copy()



# Adding both to a list to streamline cleaning each df at the same time

dfs = [train, test]

dfs_names = ['Train','Test']
# New thing I came across - Pandas profiling

#train_original.profile_report()
# Regex to get titles (start with space and end with full stop)

for dataset in [train,test]:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    

# Never seen pd.crosstab before but found this in the top rated Titanic notebook, very cool

pd.crosstab(train['Title'], train['Sex'])
# Replacing some rare ones with just the word 'Rare', and combining some similar ones into 'Miss'

for df in [train,test]:

    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    

for df in [train,test]:

    df['Title'] = df['Title'].replace(['Mlle','Mme','Ms'], 'Miss') # Mlle = Mademoiselle

    

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# regular expressions are something I need to properly learned, totally just cobbled this together from various stack overflow posts

import re

for df in [train,test]:

    df['Ticket Extracted'] = df['Ticket'].str.extract(r'([a-zA-Z]+)')
# How many unique values are there in each 'object' column i.e. what type of encoding to use

for i,j in enumerate(dfs):

    print(dfs_names[i]+':')

    for col in j.select_dtypes(include='object').columns:

        print(f'Unique number of {col}: {j[col].nunique()}')
label_enc = LabelEncoder()



to_encode = ['Sex','Embarked','Ticket Extracted','Title']



for df in [train, test]:

    for i in to_encode:

        df[i] = df[i].astype(str)

        df[i] = label_enc.fit_transform(df[i])

        

for df in [train,test]:

    df.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
# Getting dummy columns and dropping unwanted columns (not using this anymore, using above label encoding instead)



def get_dummies_and_drop(df,dummy_cols,drop_cols):

    '''Function to create dummy columns and drop unwanted columns

    

    Keyword arguments:

    df -- dataframe to use

    dummy_cols -- List of columns to create dummies of

    drop_cols -- List of columns to drop 

    '''

    df_ = pd.get_dummies(df,columns=dummy_cols)

    for i in to_drop:

        try:

            df_.drop(i,inplace=True,axis=1)

        except:

            pass

    return df_



to_dummy = ['Sex','Embarked','Ticket Extracted','Title']

to_drop = ['Sex','Embarked','Name','Ticket','Cabin','Title']



#train = get_dummies_and_drop(train,dummy_cols=to_dummy,drop_cols=to_drop)

#test = get_dummies_and_drop(test,dummy_cols=to_dummy,drop_cols=to_drop)
# What columns have NA values?



def cols_with_na(df,df_name):

    '''Print out columns with NA values, and save list of cols

    Also returns list for further use

    

    Keyword arguments:

    df -- dataframe to input

    df_name -- string to be printed out to aid readability

    '''

    # Which columns have NA values?

    cols_with_na = df.columns[df.isna().any()].to_list()

    print(f'{df_name} columns with NA values: {cols_with_na}')

    return(cols_with_na)



train_na_cols = cols_with_na(train,'Train')

test_na_cols = cols_with_na(test,'Test')
# Of columns with NA values, how many NAs are there?

# I.e. if not many then I can impute, but if too many then may have to drop column



def percent_na_values(df,df_name,impute_thres):

    '''Print out % of columns with NA values that is made up of NA values

    Allows determination of if imputation is possible or if column needs to be dropped

    

    Keyword arguments:

    df -- dataframe to input

    df_name -- string to be printed out to aid readability

    impute_thres -- threshold for determining whether to impute: if 25, will print "ok to impute" for cols with <= 25% na

    '''

    # Which columns have NA values?

    cols_with_na = df.columns[df.isna().any()].to_list()

    df_len = len(df)



    # For cols with na, print % of col that is na, and print 'ok to impute' if thres < inputted threshold

    for i in cols_with_na:

        count_missing = df[i].isna().sum()

        percent_missing = round(count_missing/df_len*100,1)

        print(f'Percent of {df_name} {i} missing = {percent_missing}')

        if percent_missing <= impute_thres:

            print(f'Ok to impute {i} (below chosen {impute_thres}% impute threshold)')

        else:

            print(f'Don\'t impute {i}')

        

percent_na_values(train,'Train',impute_thres=25)

print('')

percent_na_values(test,'Test',impute_thres=25)
# Deciding against using this for now



def multivariate_imputer(df):

    """Use sklearn's multivariate imputer to impute in a more sophisticated way than just the median"""

    imp = IterativeImputer(max_iter=20, random_state=0,min_value=0)

    imp.fit(df)

    df_imputed = imp.transform(df)

    df_imputed = pd.DataFrame(df_imputed,columns=df.columns)

    return df_imputed



train = multivariate_imputer(train)

test = multivariate_imputer(test)
imputer = SimpleImputer(strategy='most_frequent')



def impute_df(df):

    df_imputed = pd.DataFrame(imputer.fit_transform(df))

    df_imputed.columns = df.columns

    df = df_imputed.copy() # to get back to normal naming convention

    return df



#train = impute_df(train)

#test = impute_df(test)



print(f'Total NA in train: {train.isna().sum().sum()}')

print(f'Total NA in test: {test.isna().sum().sum()}')
# qcut to create age and fare bins

def bin_maker(df, col_name, num_bins):

    '''Input dataframe, the name of the column and the number of qcut bins'''

    df[f'{col_name} Bins'] = pd.qcut(df[col_name],q=num_bins,labels=False,duplicates='drop')



for i in [train,test]:

    for j in ['Age','Fare']:

        bin_maker(df=i,col_name=j,num_bins=10)
def threshold_explainer(df,col,num_bins):

    """Prints the min and max for each qcut threshold"""

    for i in np.arange(num_bins):

        df_ = df[df[col+' Bins']==i].copy()

        min_ = df_[col].min()

        max_ = df_[col].max()

        print(f'Min for {col} Bins group {i} is {min_} and max is {max_}')

        

threshold_explainer(train,'Age',num_bins=10)

print('')

threshold_explainer(train,'Fare',num_bins=10)
# Add "family" column

for df in [train, test]:

    df['Family'] = df['Parch'] + df['SibSp']
# Correlation matrix - maybe I should have used label encoding rather than one-hot!!

corr_matrix = train.corr()

fig = plt.figure(figsize=[15,10])

sns.heatmap(corr_matrix,annot=True)
# Dropping age, fare and passenger ID as logistic regression wants all features to be similarly scaled 

for df in [train,test]:

    df.drop(['Age','Fare'],inplace=True,axis=1)
# Making X and y

X = train.copy()

X.drop('Survived',inplace=True,axis=1)

y = train['Survived'].copy()



# Train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
def gridsearch(X,y,model,grid,cv):

    '''Gridsearch function to perform a gridsearch with different models

    And then return the best parameters for use

    

    Keyword arguments:

    model -- model to use i.e. XGB

    grid -- dictionary of parameters to try out

    cv -- size of cross validation

    '''

    CV = GridSearchCV(estimator=model,param_grid=grid,cv=cv)

    CV.fit(X, y)

    print(CV.best_params_)

    print('Best parameters returned for use')

    return(CV.best_params_)
xgb = XGBClassifier()

xgb_grid = {

    'learning_rate':[0.001,0.002,0.003,0.004,0.005],

    'n_estimators':[60,70,80,90],

    'max_depth':[2,3,4],

    'subsample':[0.1,0.2,0.3,0.4]}



forest = RandomForestClassifier()

forest_grid = {

    'max_depth':[4,5,6],

    'max_leaf_nodes':[25,30,35],

    'n_estimators':[550,600,650]}



# Run the function

xgb_params = gridsearch(X,y,xgb,xgb_grid,5)

forest_params = gridsearch(X,y,forest,forest_grid,5)
xgb = XGBClassifier(learning_rate=xgb_params['learning_rate'],max_depth=xgb_params['max_depth'],

                    n_estimators=xgb_params['n_estimators'],subsample=xgb_params['subsample'])



forest = RandomForestClassifier(max_depth=forest_params['max_depth'],max_leaf_nodes=forest_params['max_leaf_nodes'],

                                n_estimators=forest_params['n_estimators'])
def cross_val(model_name,model,X,y,cv):

    '''Cross validate a model and gives scores and average score

    

    Keyword arguments:

    model_name -- string of the name, for printing out

    model -- model i.e. xgb, forest

    X -- data to use with no target

    y -- target

    cv -- number of cross validations

    '''

    scores = cross_val_score(model, X, y, cv=cv)

    print(f'{model_name} Scores:')

    for i in scores:

        print(round(i,2))

    print(f'Average {model_name} score: {round(scores.mean(),2)}')
cross_val('XGB',xgb,X,y,5)

cross_val('Forest',forest,X,y,5)
xgb.fit(X,y)



preds = xgb.predict(test)



test.PassengerId.astype(int)
output = pd.DataFrame({'PassengerId': test.PassengerId.astype(int), 'Survived': preds.astype(int)})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")