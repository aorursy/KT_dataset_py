#import linear algebra and data manipulation libraries

import numpy as np

import pandas as pd



#import standard visualization

import matplotlib.pyplot as plt

import seaborn as sns



#import machine learning

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



import xgboost



from sklearn.model_selection import train_test_split #split

from sklearn.metrics import accuracy_score, roc_auc_score #metrics
# data load

#df = pd.read_csv('/kaggle/input/devrepublik03/training_set.csv') # set for training-testing

df = pd.read_csv('/kaggle/input/devrepublik03/validation_set.csv') # set for final validation
df
#find percentage of missing values for each column

missing_values = df.isnull().mean()*100



missing_values.sum()
#balance and deposit



b_df = pd.DataFrame()

b_df['balance_yes'] = (df[df['deposit'] == 'yes'][['deposit','balance']].describe())['balance']

b_df['balance_no'] = (df[df['deposit'] == 'no'][['deposit','balance']].describe())['balance']



b_df
def get_dummy_from_bool(row, column_name):

    ''' Returns 0 if value in column_name is no, returns 1 if value in column_name is yes'''

    return 1 if row[column_name] == 'yes' else 0



def get_correct_values(row, column_name, threshold, df):

    ''' Returns mean value if value in column_name is above threshold'''

    if row[column_name] <= threshold:

        return row[column_name]

    else:

        mean = df[df[column_name] <= threshold][column_name].mean()

        return mean



def clean_data(df):

    '''

    INPUT

    df - pandas dataframe containing bank marketing campaign dataset

    

    OUTPUT

    df - cleaned dataset:

    1. columns with 'yes' and 'no' values are converted into boolean variables;

    2. categorical columns are converted into dummy variables;

    3. drop irrelevant columns.

    4. impute incorrect values

    '''

    

    cleaned_df = df.copy()

    

    #convert columns containing 'yes' and 'no' values to boolean variables and drop original columns

    bool_columns = ['default', 'housing', 'loan']

    for bool_col in bool_columns:

        cleaned_df[bool_col + '_bool'] = df.apply(lambda row: get_dummy_from_bool(row, bool_col),axis=1)

    

    cleaned_df = cleaned_df.drop(columns = bool_columns)

    

    #convert categorical columns to dummies

    cat_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

    

    for col in  cat_columns:

        cleaned_df = pd.concat([cleaned_df.drop(col, axis=1),

                                pd.get_dummies(cleaned_df[col], prefix=col, prefix_sep='_',

                                               drop_first=True, dummy_na=False)], axis=1)

    

    #drop irrelevant columns

    cleaned_df = cleaned_df.drop(columns = ['pdays'])

    

    #impute incorrect values and drop original columns

    cleaned_df['campaign_cleaned'] = df.apply(lambda row: get_correct_values(row, 'campaign', 34, cleaned_df),axis=1)

    cleaned_df['previous_cleaned'] = df.apply(lambda row: get_correct_values(row, 'previous', 34, cleaned_df),axis=1)

    

    cleaned_df = cleaned_df.drop(columns = ['campaign', 'previous'])

    

    return cleaned_df
#clean the dataset

cleaned_df = clean_data(df)

cleaned_df.head()
X = cleaned_df.drop(columns = 'deposit_bool')

y = cleaned_df[['deposit_bool']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
xgb = xgboost.XGBClassifier(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,

                           colsample_bytree=1, max_depth=7)

xgb.fit(X_train,y_train.squeeze().values)



#calculate and print scores for the model for top 15 features

y_train_preds = xgb.predict(X_train)

y_test_preds = xgb.predict(X_test)



print('XGB accuracy score for train: %.3f: test: %.3f' % (

        accuracy_score(y_train, y_train_preds),

        accuracy_score(y_test, y_test_preds)))
y_proba
y_proba = xgb.predict_proba(X_test)

auc_score = roc_auc_score(y_test, y_proba[:,1])

auc_score
model = GradientBoostingClassifier(

    n_estimators=150,

    learning_rate=0.1,

    min_samples_split=4, 

    min_samples_leaf=2, 

    min_weight_fraction_leaf=0.0, 

    max_depth=4

)

model.fit(X_train,y_train)







y_proba = model.predict_proba(X_test)

auc = roc_auc_score(y_test, y_proba[:,1])

auc
subm = pd.DataFrame() # create an empty DF for final submissiob

subm['deposit'] = model.predict_proba(cleaned_df)[:,1] # create a new column that holds probability of first (1) class

subm.reset_index(drop=False, inplace=True) # duplicate index as a columns

subm.to_csv('my_submission.csv', index=False) # save as csv file
cleaned_df
subm.head()