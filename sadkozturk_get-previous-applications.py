# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook as tqdm
# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
#summarize all categorical variables: function for categorical variables and cat categorical with label encoded (0-1)
def catSummary(df, number_of_classes = 10, num = False):
    
    #summarize all categorical variables
    import pandas as pd
    import numpy as np
    
    var_count = 0
    vars_more_classes = []
    
    for var in df:
        if num == False:
            if df[var].dtype == 'object':
                if len(list(df[var].unique())) <= number_of_classes:
                    print(pd.DataFrame({var: df[var].value_counts(), "RATIO": 100 * df[var].value_counts() / len(df)}),end = "\n\n\n")
                    var_count += 1
                else:
                    vars_more_classes.append(df[var].name)
        else:
            if len(list(df[var].unique())) <= number_of_classes:
                print(pd.DataFrame({var: df[var].value_counts(), "RATIO": 100 * df[var].value_counts() / len(df)}),end = "\n\n\n")
                var_count += 1
            else:
                vars_more_classes.append(df[var].name)
                
    print('%d categorical variables were described' % var_count, end = "\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end = "\n\n")
    print(vars_more_classes)
def missingTable(df, by = False, number_of_missing = 0, missing_ratio = 20):
    
    """
    by: specifying sorting method: False, n_miss, ratio
    
    """
    
    n_miss = df.isnull().sum().sort_values(ascending = False)
    ratio = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    missing_df = pd.concat([n_miss, ratio], axis = 1, keys = ['n_miss', 'ratio'])
    
    if by == False:
        return missing_df
    
    elif by == "n_miss":
        
        varCountMissing = missing_df[missing_df["n_miss"]>number_of_missing].index.value_counts().sum()
        print(varCountMissing,"variables have at least",number_of_missing + 1, "value",end = "\n\n\n")
        return missing_df[missing_df["n_miss"]>number_of_missing]
    
    elif by == "ratio":

        varCountRatio = missing_df[missing_df["ratio"]>missing_ratio].index.value_counts().sum()
        print(varCountRatio,"variables have at least",missing_ratio, "% missing value",end = "\n\n\n")
        return missing_df[missing_df["ratio"]>missing_ratio]
    
def hasVarOutliers(df):
    for var in df:
        if var != "SK_ID_CURR":
            if df[var].dtype != 'object':
                if len(list(df[var].unique())) > 10:
                    
                    Q1 = df[var].quantile(0.01)
                    Q3 = df[var].quantile(0.99)
                    IQR = Q3-Q1
                    lower = Q1- 1.5*IQR
                    upper = Q3 + 1.5*IQR
                    
                    if df[(df[var] > upper)].any(axis=None):
                        print(var,"----HAS OUTLIERS")
def fillOutliers(data):
    df = data.copy()
    for var in df:
        if var != "SK_ID_CURR":
            if df[var].dtype != 'object':
                if len(list(df[var].unique())) > 10:
                    
                    Q1 = df[var].quantile(0.01)
                    Q3 = df[var].quantile(0.99)
                    IQR = Q3-Q1
                    lower = Q1- 1.5*IQR
                    upper = Q3 + 1.5*IQR
                    
                    if df[(df[var] > upper)].any(axis=None):
                        maks = df[var].max()
                        minn = df[var].min()
                        
                        df[var][(df[var] > (upper))] = upper       
                        #df[var][(df[var] < (lower))] = lower 
                        print(var,"Differences Between Max and Upper:",str(maks-upper),"upper:",upper)
                        #print(var,"Differences Between Min and lower",str(minn-lower))
    return df                             
# one-hot encoding of categorical variables
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns = categorical_columns, dummy_na = nan_as_category)
    cat_columns = [c for c in df.columns if c not in original_columns]
    return df, cat_columns
""" Process previous_application.csv and return a pandas dataframe. """
previous_application = pd.read_csv('../input/home-credit-default-risk/previous_application.csv')
print('previous_application data shape: ', previous_application.shape)
previous_application.head()
catSummary(previous_application)
missingTable(previous_application)
hasVarOutliers(previous_application)
fillOutliers(previous_application)
prev, prev_cat_cols = one_hot_encoder(previous_application)
prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
# Add feature: value ask / value received percentage

prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
prev['NEW_CREDIT_TO_ANNUITY_RATIO'] = prev['AMT_CREDIT']/prev['AMT_ANNUITY']
prev['NEW_DOWN_PAYMENT_TO_CREDIT'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_CREDIT']
prev['NEW_TOTAL_PAYMENT'] = prev['AMT_ANNUITY'] * prev['CNT_PAYMENT']
prev['NEW_TOTAL_PAYMENT_TO_AMT_CREDIT'] = prev['NEW_TOTAL_PAYMENT'] / prev['AMT_CREDIT']
# Innterest ratio previous application (simplified)

prev['SIMPLE_INTERESTS'] = (prev['NEW_TOTAL_PAYMENT']/prev['AMT_CREDIT'] - 1)/prev['CNT_PAYMENT']
# Previous applications numeric features
num_aggregations = {}
num_cols = prev.select_dtypes(exclude=['object']) 
num_cols.drop(['SK_ID_PREV', 'SK_ID_CURR'], axis= 1,inplace = True)
for num in num_cols:
    num_aggregations[num] = ['min', 'max', 'mean', 'var','sum']
prev_agg.shape
# Previous applications numeric features
cat_aggregations = {}
for cat in prev_cat_cols:
    cat_aggregations[cat] = ['mean']
prev_agg.shape
prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
prev_agg.shape
features_with_small_variance = prev_agg.columns[(prev_agg.std(axis = 0) < .1).values]
print(len(features_with_small_variance))
prev_agg[features_with_small_variance].describe().T
print(prev_agg.shape)
prev_agg.drop(features_with_small_variance, axis = 1, inplace = True)
print(prev_agg.shape)
prev_agg.info()




