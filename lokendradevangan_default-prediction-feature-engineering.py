# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Memory management
import gc 

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

import os
dataset= pd.read_csv('../input/application_train.csv')
Path='../input/' 

##Initial data understanding 

##test dataset which is to be predicted
test=pd.read_csv("../input/application_test.csv")
#Checking type of the variables REGION_RATING_CLIENT REGION_RATING_CLIENT_W_CITY
check =dataset[["REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY"]]
for col in check:
    if check[col].dtype != 'object':
        print (col)
dataset.REGION_RATING_CLIENT.dtype   

dataset["REGION_RATING_CLIENT"] = dataset["REGION_RATING_CLIENT"].astype('object')
dataset["REGION_RATING_CLIENT_W_CITY"] = dataset["REGION_RATING_CLIENT_W_CITY"].astype('object')

test["REGION_RATING_CLIENT"] = test["REGION_RATING_CLIENT"].astype('object')
test["REGION_RATING_CLIENT_W_CITY"] = test["REGION_RATING_CLIENT_W_CITY"].astype('object')
dataset.groupby(['REGION_RATING_CLIENT']).TARGET.mean()
dataset.groupby(['REGION_RATING_CLIENT_W_CITY']).TARGET.mean()
#reducing the unique values in occupation by grouping by skill level. This grouping can differ based on more information about each occupation
dataset['NAME_TYPE_SUITE'].replace({'Children':'Family',
                                    'Group of people':'Other',
                                    'Other_A':'Other',
                                    'Other_B':'Other',
                                    'Spouse, partner':'Family'},inplace=True)

test['NAME_TYPE_SUITE'].replace({'Children':'Family',
                                    'Group of people':'Other',
                                    'Other_A':'Other',
                                    'Other_B':'Other',
                                    'Spouse, partner':'Family'},inplace=True)
 
dataset['NAME_EDUCATION_TYPE'].replace({'Academic degree':'Higher education '},inplace=True)
test['NAME_EDUCATION_TYPE'].replace({'Academic degree':'Higher education '},inplace=True)
#reducing the unique values in occupation by grouping by skill level. This grouping can differ based on more information about each occupation
dataset['OCCUPATION_TYPE'].replace({'High skill tech staff':'High_Skill',
                                    'Managers':'High_Skill',
                                    'Accountants':'High_Med_Skill',
                                    'HR staff':'High_Med_Skill',
                                    'Core staff':'Med_Skill',
                                   'Cooking staff':'Med_Skill',
                                    'Realty agents':'Med_Skill',
                                    'Sales staff':'Med_Skill',
                                    'IT staff':'High_Med_Skill',
                                    'Medicine staff':'High_Med_Skill',
                                    'Secretaries':'Med_Skill',
                                    'Security staff':'Med_Skill',
                                    'Cleaning staff':'Low_Skill',
                                      'Laborers':'Low_Skill',
                                      'Low-skill Laborers':'Low_Skill',
                                      'Cleaning staff':'Low_Skill',
                                    'Waiters/barmen staff':'Low_Skill',
                                    'Private service staff':'Low_Skill',
                                    'Drivers':'Med_Skill'
                                   },inplace=True)
test['OCCUPATION_TYPE'].replace({'High skill tech staff':'High_Skill',
                                    'Managers':'High_Skill',
                                    'Accountants':'High_Med_Skill',
                                    'HR staff':'High_Med_Skill',
                                    'Core staff':'Med_Skill',
                                   'Cooking staff':'Med_Skill',
                                    'Realty agents':'Med_Skill',
                                    'Sales staff':'Med_Skill',
                                    'IT staff':'High_Med_Skill',
                                    'Medicine staff':'High_Med_Skill',
                                    'Secretaries':'Med_Skill',
                                    'Security staff':'Med_Skill',
                                    'Cleaning staff':'Low_Skill',
                                      'Laborers':'Low_Skill',
                                      'Low-skill Laborers':'Low_Skill',
                                      'Cleaning staff':'Low_Skill',
                                    'Waiters/barmen staff':'Low_Skill',
                                    'Private service staff':'Low_Skill',
                                    'Drivers':'Med_Skill'
                                   },inplace=True)
#grouping
dataset['NAME_INCOME_TYPE'].replace({'Businessman':'Other','Student':'Other','Maternity leave':'Other'},inplace=True)
test['NAME_INCOME_TYPE'].replace({'Businessman':'Other','Student':'Other','Maternity leave':'Other'},inplace=True)
app_train=dataset.copy()
app_test=test.copy()
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in dataset:
    if dataset[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(dataset[col].unique())) <= 2:
            # Train on the training data
            le.fit(dataset[col])
            # Transform both training and testing data
            app_train[col] = le.transform(dataset[col])
            app_test[col] = le.transform(test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)
gc.enable()
del dataset, test
gc.collect()
app_train.head()
app_train= pd.get_dummies(app_train)
app_test= pd.get_dummies(app_test)
# Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
#app_train['DAYS_EMPLOYED'].replace({0: 1}, inplace = True)
app_train.fillna(app_train.median(),inplace = True)
app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
# Replace the anomalous values with nan
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
#app_test['DAYS_EMPLOYED'].replace({0: 1}, inplace = True)
app_test.fillna(app_test.median(),inplace = True)

app_test.head()
# Create an id change flag column
app_train['ID_CHANGE_LST3M'] = app_train["DAYS_ID_PUBLISH"] <=-90
# Create an reg change flag column
app_train['REG_CHANGE_LST3M'] = app_train["DAYS_REGISTRATION"] <=-90

# Create an reg change flag column
app_test['ID_CHANGE_LST3M'] = app_test["DAYS_ID_PUBLISH"] <=-90
app_test['REG_CHANGE_LST3M'] = app_test["DAYS_REGISTRATION"] <=-90

#replace all  NaN in the var_list with zero
Var_List=('OBS_30_CNT_SOCIAL_CIRCLE','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE',
        'AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK',
         'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR')
def missing_val_replace(data,Var_List):
    for col in data:
        for i in Var_List:
            if col==i:
                data[col].fillna(0)
                print (col)
    return data
app_train=missing_val_replace(app_train,Var_List) 
#replace all other NaN with median values

app_test=missing_val_replace(app_test,Var_List) 
#replace all other NaN with median values
app_train=app_train.fillna(app_train.median)
app_test=app_test.fillna(app_test.median)

app_train.dtypes.value_counts()
app_test.dtypes.value_counts()

app_train['CREDIT_INCOME_PERCENT'] = app_train['AMT_CREDIT'] / app_train['AMT_INCOME_TOTAL']
app_train['ANNUITY_INCOME_PERCENT'] = app_train['AMT_ANNUITY'] / app_train['AMT_INCOME_TOTAL']
app_train['CREDIT_TERM'] = app_train['AMT_ANNUITY'] / app_train['AMT_CREDIT']
app_train['DAYS_EMPLOYED_PERCENT'] = app_train['DAYS_EMPLOYED'] / app_train['DAYS_BIRTH']
app_train['INCOME_AGE_RATIO'] = app_train['AMT_INCOME_TOTAL'] / app_train['DAYS_BIRTH']

app_train['AMT_CR_AGE_RATIO'] = app_train['AMT_CREDIT'] / app_train['DAYS_BIRTH']
app_train['CREDIT_AMT_GDS_PERCENT'] = app_train['AMT_CREDIT'] / app_train['AMT_GOODS_PRICE']
app_train['AMT_GDS_INCOME_PERCENT'] = app_train['AMT_GOODS_PRICE'] / app_train['AMT_INCOME_TOTAL']
#app_train['AMT_CR_EMP_RATIO'] = app_train['AMT_CREDIT'] / app_train['DAYS_EMPLOYED']

app_test['CREDIT_INCOME_PERCENT'] = app_test['AMT_CREDIT'] / app_test['AMT_INCOME_TOTAL']
app_test['ANNUITY_INCOME_PERCENT'] = app_test['AMT_ANNUITY'] / app_test['AMT_INCOME_TOTAL']
app_test['CREDIT_TERM'] = app_test['AMT_ANNUITY'] / app_test['AMT_CREDIT']
app_test['DAYS_EMPLOYED_PERCENT'] = app_test['DAYS_EMPLOYED'] / app_test['DAYS_BIRTH']

app_test['INCOME_AGE_RATIO'] = app_test['AMT_INCOME_TOTAL'] / app_test['DAYS_BIRTH']
app_test['AMT_CR_AGE_RATIO'] = app_test['AMT_CREDIT'] / app_test['DAYS_BIRTH']
app_test['CREDIT_AMT_GDS_PERCENT'] = app_test['AMT_CREDIT'] / app_test['AMT_GOODS_PRICE']
app_test['AMT_GDS_INCOME_PERCENT'] = app_test['AMT_GOODS_PRICE'] / app_test['AMT_INCOME_TOTAL']
#app_test['AMT_CR_EMP_RATIO'] = app_test['AMT_CREDIT'] / app_test['DAYS_EMPLOYED']

#bureau.csv
path="../input/"
bureau=pd.read_csv("../input/bureau.csv")
bureau.head()
def count_categorical(df, group_var, df_name):
    """Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns

    
    Return
    --------
    categorical : dataframe
        A dataframe with counts and normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`.
        
    """
    
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes(include=['object']))

    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names
    
    return categorical
def agg_numeric(df, group_var, df_name):
    """Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.
    
    Parameters
    --------
        df (dataframe): 
            the dataframe to calculate the statistics on
        group_var (string): 
            the variable by which to group df
        df_name (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            the statistics (mean, min, max, sum; currently supported) calculated. 
            The columns are also renamed to keep track of features created.
    
    """
    # Remove id variables other than grouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)
            
    group_ids = df[group_var]
    numeric_df = df.select_dtypes(include=['number'])
    numeric_df[group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

    # Need to create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    return agg
# Counts of each type of caterigical variable bureau
bureau_counts = count_categorical(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_counts.head()
bureau_agg = agg_numeric(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_agg.head()

# Read in bureau balance
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
bureau_balance.head()
# Calculate summary statistics for each `SK_ID_BUREAU` 
bureau_balance_agg = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
bureau_balance_agg.head()
# Counts of each type of status for each previous loan
bureau_balance_counts = count_categorical(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
bureau_balance_counts.head()
app_test.head()
# Dataframe grouped by the loan
bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')

# Merge to include the SK_ID_CURR
bureau_by_loan = bureau_by_loan.merge(bureau[['SK_ID_BUREAU', 'SK_ID_CURR']], on = 'SK_ID_BUREAU', how = 'left')

bureau_by_loan.head()
#summary by cust id
bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(['SK_ID_BUREAU'],1), group_var = 'SK_ID_CURR', df_name = 'client')
bureau_balance_by_client.head()
del bureau_by_loan
# Read in previous_application 
previous_application = pd.read_csv('../input/previous_application.csv')
previous_application.head()
#histogram of price by condition and brand
# Histogram
# bins = number of bar in figure
previous_application[['DAYS_LAST_DUE','DAYS_TERMINATION','DAYS_LAST_DUE_1ST_VERSION','DAYS_FIRST_DUE','DAYS_FIRST_DRAWING']].describe()
# Create an anomalous flag column
previous_application['DAYS_LAST_DUE_ANOMALY_FLG'] = previous_application["DAYS_LAST_DUE"] == 365243
previous_application['DAYS_TERMINATION_ANOMALY_FLG'] = previous_application["DAYS_TERMINATION"] == 365243
previous_application['DAYS_LAST_DUE_1ST_VERSION_ANOMALY_FLG'] = previous_application["DAYS_LAST_DUE_1ST_VERSION"] == 365243
previous_application['DAYS_FIRST_DUE_ANOMALY_FLG'] = previous_application["DAYS_FIRST_DUE"] == 365243
previous_application['DAYS_FIRST_DRAWING_ANOMALY_FLG'] = previous_application["DAYS_FIRST_DRAWING"] == 365243
# Replace the anomalous values with nan
previous_application['DAYS_LAST_DUE'].replace({365243: np.nan}, inplace = True)
previous_application['DAYS_TERMINATION'].replace({365243: np.nan}, inplace = True)
previous_application['DAYS_LAST_DUE_1ST_VERSION'].replace({365243: np.nan}, inplace = True)
previous_application['DAYS_FIRST_DUE'].replace({365243: np.nan}, inplace = True)
previous_application['DAYS_FIRST_DRAWING'].replace({365243: np.nan}, inplace = True)

##reducing dimenstions in categorical variable
previous_application['PRODUCT_COMBINATION'].replace({'Cash Street: high':'Cash_Street',
                                                     'Cash Street: low':'Cash_Street',
                                                     'Cash Street: middle':'Cash_Street',
                                                       'Cash X-Sell: high':'Cash_XSell',
                                                      'Cash X-Sell: middle':'Cash_XSell',
                                                      'Cash X-Sell: low':'Cash_XSell',
                                                     'POS household with interest':'POS_Interest',                                                       
'POS household without interest':'POS_No_Interest',     
'POS industry with interest': 'POS_Interest'   ,    
'POS industry without interest': 'POS_No_Interest' ,    
'POS mobile with interest':  'POS_Interest',        
'POS mobile without interest':   'POS_No_Interest',    
'POS other with interest':    'POS_Interest'    ,   
'POS others without interest':  'POS_No_Interest'      
                                                     
                                    },inplace=True)
previous_application['NAME_CONTRACT_TYPE'].replace({'XNA':'Unknown'},inplace=True)
previous_application.groupby(['NAME_CONTRACT_STATUS']).SK_ID_PREV.count()
previous_application['NAME_PORTFOLIO'].replace({'Cars':'Cards'
                                    },inplace=True)

previous_application['CHANNEL_TYPE'].replace({'Car dealer':'Other',
                                    'Channel of corporate sales':'Other'
                            },inplace=True)
previous_application.groupby(['CHANNEL_TYPE']).SK_ID_PREV.count()
previous_application['CODE_REJECT_REASON'].replace({'SYSTEM':'Other',
                                    'VERIF':'Other',
                                    'XNA':'Other',
                                    'VERIF':'Other'
                            },inplace=True)

previous_application['NAME_SELLER_INDUSTRY'].replace({'Auto technology':'Other',
                                    'Jewelry':'Other',
                                    'MLM partners':'Other',
                                    'Tourism':'Other'
                            },inplace=True)
previous_application.groupby(['NAME_SELLER_INDUSTRY']).SK_ID_PREV.count()
previous_application['NAME_CASH_LOAN_PURPOSE'].replace({'Building a house or an annex':'House',
                                                    'Buying a garage':'House',
                                                    'Buying a home':'House',
                                                    'Buying a holiday home / land':'House',
                                                    'Repairs':'House',
                                                    'Buying a new car':'LifeStyle',
                                                   'Buying a used car':'LifeStyle',
                                                   'Car repairs':'LifeStyle',
                                                   'Furniture':'LifeStyle',
                                                   'Hobby':'LifeStyle',
                                                   'Journey':'LifeStyle',
                                                   'Wedding / gift / holiday':'LifeStyle',
                                                       'Purchase of electronic equipment':'LifeStyle',
                                                        'Urgent needs':'Needs',
                                                        'Refusal to name the goal':'Needs',
                                                        'Money for a third person':'Needs',
                                                        'Gasification / water supply':'Needs',
                                                        'Medicine':'Needs',
                                                        'Payments on other loans':'Needs',
                                                        'Everyday expenses':'Needs',
                                                        'Business development':'Needs'                                                       
                                                       },inplace=True)
previous_application.groupby(['NAME_CASH_LOAN_PURPOSE']).SK_ID_PREV.count()
previous_application['NAME_TYPE_SUITE'].replace({'Children':'Family',
                                    'Group of people':'Other',
                                    'Other_A':'Other',
                                    'Other_B':'Other',
                                    'Spouse, partner':'Family'},inplace=True)
previous_application.groupby(['NAME_TYPE_SUITE']).SK_ID_PREV.count()
#drop categorical variables which are captured by other variables and does not makes sense to keep it 
#drop name goods category as it is broadly captured by Industry variable
previous_application=previous_application.drop(['NAME_GOODS_CATEGORY','WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START'],1)
# Calculate value counts for each categorical column
previous_counts = count_categorical(previous_application, group_var = 'SK_ID_CURR', df_name = 'previous_loans')
previous_counts.head()
# Calculate aggregate statistics for each numeric column
previous_agg = agg_numeric(previous_application.drop( ['SK_ID_PREV'],1), group_var = 'SK_ID_CURR', df_name = 'previous_loans')
previous_agg.head()
# Remove variables to free memory

del bureau, bureau_balance,previous_application
previous_agg['previous_loans_CREDIT_TERM'] = previous_agg['previous_loans_AMT_ANNUITY_sum'] / previous_agg[ 'previous_loans_AMT_CREDIT_sum']

previous_agg['previous_loans_CREDIT_AMT_GDS_PERCENT'] = previous_agg['previous_loans_AMT_GOODS_PRICE_sum'] / previous_agg['previous_loans_AMT_CREDIT_sum']

previous_agg['previous_loans_AMT_APPLICATION_PERCENT'] = previous_agg['previous_loans_AMT_APPLICATION_sum'] / previous_agg['previous_loans_AMT_CREDIT_sum']

previous_agg['previous_loans_AMT_DOWN_PAYMENT_PERCENT'] = previous_agg['previous_loans_AMT_DOWN_PAYMENT_sum'] / previous_agg['previous_loans_AMT_CREDIT_sum']

previous_agg['previous_loans_AMT_APPLICATION_range'] = previous_agg['previous_loans_AMT_APPLICATION_max']-previous_agg['previous_loans_AMT_APPLICATION_min']

previous_agg['previous_loans_AMT_GOODS_PRICE_range'] = previous_agg['previous_loans_AMT_GOODS_PRICE_max']-previous_agg['previous_loans_AMT_GOODS_PRICE_min']

previous_agg[['previous_loans_DAYS_LAST_DUE_mean','previous_loans_DAYS_LAST_DUE_min','previous_loans_DAYS_LAST_DUE_max']].describe()
#'DAYS_TERMINATION','DAYS_LAST_DUE_1ST_VERSION','DAYS_FIRST_DUE','DAYS_FIRST_DRAWING']].
def agg_numeric(df, parent_var, df_name):
    """
    Groups and aggregates the numeric values in a child dataframe
    by the parent variable.
    
    Parameters
    --------
        df (dataframe): 
            the child dataframe to calculate the statistics on
        parent_var (string): 
            the parent variable used for grouping and aggregating
        df_name (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated by the `parent_var` for 
            all numeric columns. Each observation of the parent variable will have 
            one row in the dataframe with the parent variable as the index. 
            The columns are also renamed using the `df_name`. Columns with all duplicate
            values are removed. 
    
    """
    
    # Remove id variables other than grouping variable
    for col in df:
        if col != parent_var and 'SK_ID' in col:
            df = df.drop(columns = col)
            
    # Only want the numeric variables
    parent_ids = df[parent_var].copy()
    numeric_df = df.select_dtypes('number').copy()
    numeric_df[parent_var] = parent_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(parent_var).agg(['count', 'mean', 'max', 'min', 'sum'])

    # Need to create new column names
    columns = []

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        if var != parent_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))
    
    agg.columns = columns
    
    # Remove the columns with all redundant values
    _, idx = np.unique(agg, axis = 1, return_index=True)
    agg = agg.iloc[:, idx]
    
    return agg

def agg_categorical(df, parent_var, df_name):
    """
    Aggregates the categorical features in a child dataframe
    for each observation of the parent variable.
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    parent_var : string
        The variable by which to group and aggregate the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns

    
    Return
    --------
    categorical : dataframe
        A dataframe with aggregated statistics for each observation of the parent_var
        The columns are also renamed and columns with duplicate values are removed.
        
    """
    
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('category'))

    # Make sure to put the identifying id on the column
    categorical[parent_var] = df[parent_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(parent_var).agg(['sum', 'count', 'mean'])
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['sum', 'count', 'mean']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names
    
    # Remove duplicate columns by values
    _, idx = np.unique(categorical, axis = 1, return_index = True)
    categorical = categorical.iloc[:, idx]
    
    return categorical

def aggregate_client(df, group_vars, df_names):
    """Aggregate a dataframe with data at the loan level 
    at the client level
    
    Args:
        df (dataframe): data at the loan level
        group_vars (list of two strings): grouping variables for the loan 
        and then the client (example ['SK_ID_PREV', 'SK_ID_CURR'])
        names (list of two strings): names to call the resulting columns
        (example ['cash', 'client'])
        
    Returns:
        df_client (dataframe): aggregated numeric stats at the client level. 
        Each client will have a single row with all the numeric data aggregated
    """
    
    # Aggregate the numeric columns
    df_agg = agg_numeric(df, parent_var = group_vars[0], df_name = df_names[0])
    
    # If there are categorical variables
    if any(df.dtypes == 'category'):
    
        # Count the categorical columns
        df_counts = agg_categorical(df,group_vars[0], df_name = df_names[0])

        # Merge the numeric and categorical
        df_by_loan = df_counts.merge(df_agg, on = group_vars[0], how = 'outer')

        gc.enable()
        del df_agg, df_counts
        gc.collect()

        # Merge to get the client id in dataframe
        df_by_loan = df_by_loan.merge(df[[group_vars[0], group_vars[1]]], on = group_vars[0], how = 'left')

        # Remove the loan id
        df_by_loan = df_by_loan.drop(columns = [group_vars[0]])

        # Aggregate numeric stats by column
        df_by_client = agg_numeric(df_by_loan, parent_var = group_vars[1], df_name = df_names[1])

        
    # No categorical variables
    else:
        # Merge to get the client id in dataframe
        df_by_loan = df_agg.merge(df[[group_vars[0], group_vars[1]]], on = group_vars[0], how = 'left')
        
        gc.enable()
        del df_agg
        gc.collect()
        
        # Remove the loan id
        df_by_loan = df_by_loan.drop(columns = [group_vars[0]])
        
        # Aggregate numeric stats by column
        df_by_client = agg_numeric(df_by_loan, parent_var = group_vars[1], df_name = df_names[1])
        
    # Memory management
    gc.enable()
    del df, df_by_loan
    gc.collect()

    return df_by_client
import sys

def return_size(df):
    """Return size of dataframe in gigabytes"""
    return round(sys.getsizeof(df) / 1e9, 2)

def convert_types(df, print_info = False):
    
    original_memory = df.memory_usage().sum()
    
    # Iterate through each column
    for c in df:
        
        # Convert ids and booleans to integers
        if ('SK_ID' in c):
            df[c] = df[c].fillna(0).astype(np.int32)
            
        # Convert objects to category
        elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype('category')
        
        # Booleans mapped to integers
        elif list(df[c].unique()) == [1, 0]:
            df[c] = df[c].astype(bool)
        
        # Float64 to float32
        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)
            
        # Int64 to int32
        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)
        
    new_memory = df.memory_usage().sum()
    
    if print_info:
        print(f'Original Memory Usage: {round(original_memory / 1e9, 2)} gb.')
        print(f'New Memory Usage: {round(new_memory / 1e9, 2)} gb.')
        
    return df
cash = pd.read_csv('../input/POS_CASH_balance.csv')
cash = convert_types(cash, print_info=True)
cash.head()
cash_by_client = aggregate_client(cash, group_vars = ['SK_ID_PREV', 'SK_ID_CURR'], df_names = ['cash', 'client'])
cash_by_client.head()

credit = pd.read_csv('../input/credit_card_balance.csv')
credit = convert_types(credit, print_info = True)
credit.head()
credit_by_client = aggregate_client(credit, group_vars = ['SK_ID_PREV', 'SK_ID_CURR'], df_names = ['credit', 'client'])
credit_by_client.head()
installments = pd.read_csv('../input/installments_payments.csv')
installments = convert_types(installments, print_info = True)
installments.head()
installments_by_client = aggregate_client(installments, group_vars = ['SK_ID_PREV', 'SK_ID_CURR'], df_names = ['installments', 'client'])
installments_by_client.head()

app_train = app_train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
app_test = app_test.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
app_train = app_train.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')
app_test = app_test.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')


del bureau_agg,bureau_counts
app_test.head()

app_train = app_train.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')
app_test = app_test.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')


del bureau_balance_by_client

app_train = app_train.merge(previous_counts, on = 'SK_ID_CURR', how = 'left')
app_test = app_test.merge(previous_counts, on = 'SK_ID_CURR', how = 'left')

del previous_counts

app_train = app_train.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')
app_test = app_test.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')
app_test.head()



app_train = app_train.merge(cash_by_client, on = 'SK_ID_CURR', how = 'left')
app_test = app_test.merge(cash_by_client, on = 'SK_ID_CURR', how = 'left')

app_train = app_train.merge(credit_by_client, on = 'SK_ID_CURR', how = 'left')
app_test = app_test.merge(credit_by_client, on = 'SK_ID_CURR', how = 'left')

app_train = app_train.merge(installments_by_client, on = 'SK_ID_CURR', how = 'left')
app_test = app_test.merge(installments_by_client, on = 'SK_ID_CURR', how = 'left')

del previous_agg,cash_by_client,credit_by_client,installments_by_client


app_train = app_train.merge(previous_counts, on = 'SK_ID_CURR', how = 'left')
app_test = app_test.merge(previous_counts, on = 'SK_ID_CURR', how = 'left')

del previous_counts

app_train = app_train.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')
app_test = app_test.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')
app_test.head()



app_train = app_train.merge(cash_by_client, on = 'SK_ID_CURR', how = 'left')
app_test = app_test.merge(cash_by_client, on = 'SK_ID_CURR', how = 'left')

app_train = app_train.merge(credit_by_client, on = 'SK_ID_CURR', how = 'left')
app_test = app_test.merge(credit_by_client, on = 'SK_ID_CURR', how = 'left')

app_train = app_train.merge(installments_by_client, on = 'SK_ID_CURR', how = 'left')
app_test = app_test.merge(installments_by_client, on = 'SK_ID_CURR', how = 'left')

del previous_agg,cash_by_client,credit_by_client,installments_by_client
print('Installments by client shape: ', app_train.shape)
gc.enable()

gc.collect()