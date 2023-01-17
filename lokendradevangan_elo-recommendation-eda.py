import os
print(os.listdir("../input"))
#printing datasets available in this competition
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

%matplotlib inline

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

#read train and test files into python dataframe
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Number of rows and columns in training file : ",train.shape)
print("Number of rows and columns in test file : ",test.shape)
#lets look at  first five rows of training 
train.head()
#also in test
test.head()
# Number of each type of column
train.dtypes.value_counts()
test.dtypes.value_counts()
# Number of unique classes in each object column
train.select_dtypes('int64').apply(pd.Series.nunique, axis = 0)
# Number of unique classes in each object column
test.select_dtypes('int64').apply(pd.Series.nunique, axis = 0)
plt.figure(figsize=(12,8))
sns.distplot(train["target"].values, bins=60, kde=False, color="blue")
plt.title("Distribution of Loyalty score")
plt.xlabel('Loyalty score', fontsize=12)
plt.show()
(train["target"]<-30).sum()
train.groupby("feature_1").card_id.count()
fig, ax = plt.subplots(1, 3, figsize = (15, 5))
sns.countplot(train.feature_1,ax=ax[0])
sns.countplot(train.feature_2,ax=ax[1])
sns.countplot(train.feature_3,ax=ax[2])
plt.show()
train.groupby("feature_1").target.mean()
subsetDataFrame = train[train['target'] >-30]
#subsetDataFrame.groupby("feature_1").target.mean()
fig, ax = plt.subplots(2,3, figsize = (15, 5))
sns.catplot(x="feature_1", y="target", kind="bar", data=train,ax=ax[0,0])
sns.catplot(x="feature_2", y="target", kind="bar", data=train,ax=ax[0,1])
sns.catplot(x="feature_3", y="target", kind="bar", data=train,ax=ax[0,2])
sns.catplot(x="feature_1", y="target", kind="bar", data=subsetDataFrame,ax=ax[1,0])
sns.catplot(x="feature_2", y="target", kind="bar", data=subsetDataFrame,ax=ax[1,1])
sns.catplot(x="feature_3", y="target", kind="bar", data=subsetDataFrame,ax=ax[1,2])
plt.show();

### Loyalty score =-30 has impact everywhere

fig, ax = plt.subplots(1, 3, figsize = (15, 5))
sns.boxplot(x="feature_1", y="target",showfliers=False, data=train,ax=ax[0])
sns.boxplot(x="feature_2", y="target",showfliers=False, data=train,ax=ax[1])
sns.boxplot(x="feature_3", y="target",showfliers=False, data=train,ax=ax[2])
plt.show()
train = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])

plt.figure(figsize=(20,6))
sns.countplot(train.first_active_month,color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.show()

plt.figure(figsize=(20,6))
sns.countplot(test.first_active_month,color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.show()
hist = pd.read_csv("../input/historical_transactions.csv")
hist.head()

gdf = hist.groupby("card_id")
gdf = gdf["purchase_amount"].size().reset_index()
gdf.columns = ["card_id", "num_hist_transactions"]
train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")

hist.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
#creating features using numeric variables in the history file by aggregating it at card_id level
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
        if col != parent_var and 'card_id' in col:
            df = df.drop(columns = col)
            
    # Only want the numeric variables
    parent_ids = df[parent_var].copy()
    numeric_df = df.select_dtypes('number').copy()
    numeric_df[parent_var] = parent_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(parent_var).agg([ 'mean', 'max', 'min', 'sum'])

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
import sys

def return_size(df):
    """Return size of dataframe in gigabytes"""
    return round(sys.getsizeof(df) / 1e9, 2)

def convert_types(df, print_info = False):
    
    original_memory = df.memory_usage().sum()
    
    # Iterate through each column
    for c in df:
        
        # Convert ids and booleans to integers
            
        # Convert objects to category
        if (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
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
train = convert_types(train, print_info=True)
test = convert_types(test, print_info=True)
hist=convert_types(hist, print_info=True)
# Calculate value counts for each categorical column
hist_count = agg_categorical(hist[['authorized_flag','card_id','category_1','category_3']], 'card_id', 'hist')
# Calculate aggregate statistics for each numeric column
hist_agg = agg_numeric(hist[['card_id','installments','purchase_amount','month_lag']], 'card_id', 'hist')
print('Previous aggregation shape: ', hist_agg.shape)
hist_agg.head()
hist_agg=hist_agg.drop(['hist_month_lag_sum'],axis=1)
hist_agg.head()
train = pd.merge(train, hist_agg, on="card_id", how="left")
test = pd.merge(test, hist_agg, on="card_id", how="left")
# Memory management
import gc
gc.enable()
del hist_agg, gdf
gc.collect()
hist=hist[['authorized_flag','card_id','category_1','category_3']]

hist_count.head()