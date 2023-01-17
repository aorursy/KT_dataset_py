#requires: original input only: ../input/bureau_balance.csv and ../input/bureau.csv
#provides: bureau_merged.csv, a merged-rolled-up-cleaned-numericized file.

#input parameters for this kernel
infile = '../input/bureau_balance.csv'
infile2 = '../input/bureau.csv'
outfile = 'bureau_merged.csv'
index_in = 'SK_ID_BUREAU' #to define records coming in
index_out = 'SK_ID_CURR' #to define records going out; ultimately merged with test/train data so use the same ID
drop1 = ['MONTHS_BALANCE'] #executive decision to just skip this column
cat1 = ['STATUS'] #categorical column to be one-hot encoded from infile
cat2 = ['CREDIT_CURRENCY', 'CREDIT_ACTIVE'] #categorical columns to be one-hot encoded from infile2
col_mean = ['DAYS_ENDDATE_FACT', 'DAYS_CREDIT_UPDATE', 'AMT_CREDIT_SUM_LIMIT', 'DAYS_CREDIT_ENDDATE', 'CREDIT_DAY_OVERDUE'] #when rolling up, use mean for these (why? just a guess)
col_sum = ['AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY', 'DAYS_CREDIT'] #when rolling up use sum for these; always use sum for one-hot encoded
#Utility function used in this kernel
#made specifically for Home Credit data, where 'XNA' is used for NA data
def one_hot(df, columns, abbr=False):
    """
    one-hot encode specified columns in place.

    df: dataframe
    columns: columns to encode
    abbr: true, false, or dict {col:abbr, col:abbr, ...}; if set to true, automatically determine abbreviations.  Uniqueness not guaranteed! (alpha software)
    return: {oldcol:[newcol, newcol, ...], oldcol:[newcol, newcol, ...], ...}
    """
    colmap = {}
    if type(columns) == str:
        columns = [columns] #allow using a string for a single column
    if abbr is True: 
        abbr = {}
        for col in columns:
            #make a reasonable abbraviation
            c = str(col).replace(' ', '_').replace('(', '_').replace(')', '_').replace('__', '_').replace('__', '_').replace('__', '_') 
            if c.endswith('_'):
                c=c[:-1]
            if c.startswith('_'):
                c=c[1:]
            abbr[col] = ''.join([x[0] for x in c.split('_') if len(x)])
    for col in columns:
        #one-hot encode
        d = pd.get_dummies(df[col])
        if 'XNA' in d: #drop XNA if it exists
            del d['XNA']
        else:
            del d[d.columns[-1]] #else, drop the last entry
        if abbr and col in abbr: #create new columns out of abbreviated column name and mildly-abbreviatied dummy categorical values
            d.columns = [abbr[col] + '_' + str(x).replace(' ','').replace('/','').replace(':','').replace('(','').replace(')','').replace(',','') for x in d.columns]
        else:#create new columns out of full column name and full dummy categorical values
            d.columns = [str(col) + '_' + str(x) for x in d.columns]
        df.drop(columns=col, inplace=True) #drop the original columns
        df[d.columns] = d #add the new dummy columns
        colmap[col] = d.columns #put new columns in dict so we can return it
    return colmap
import pandas as pd
import numpy as np
#import matplotlib as plt
#import seaborn as sbn
import os, sys

#cause all of cell, not just last, line to display result
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#read the file, using bureau IDs as dataframe index; AFAIK the future warning is harmless and I don't know how to get rid of it
df = pd.read_csv(infile, index_col=index_in)
df.head()
len(df)
#one-hot encode the categorical columns
col = one_hot(df, columns=cat1, abbr=True)

#drop the unwanted column
df.drop(columns=drop1, inplace=True)
df.head()
#Aggregate the columns; for infile1, they are ALL summed since they were one-hot encoded
df = df.groupby(by=index_in).agg({x:'sum' for x in df.columns})
df.head()
len(df)
#Read the second file that will be merged with the first data; AFAIK the future warning is harmless and I don't know how to get rid of it
df2 = pd.read_csv(infile2, index_col=index_in)
df2.head()
#Merge it by index
df = pd.merge(df2, df, left_index=True, right_index=True, how='left')
df.head()
len(df)
#One-hot encode categorical columns
col2 = one_hot(df, columns=cat2, abbr=True)
df.head()
#Create the aggregation dictionary for group-and-aggregate
agg = {}
for c in col:
    agg = {x:'sum' for x in col[c]}
for c in col2:
    agg.update({x:'sum' for x in col2[c]})
for c in col_mean:
    agg[c] = 'mean'
for c in col_sum:
    agg[c] = 'sum'
agg

for c in df.columns:
    if c not in agg:
        print(c)
#first put 0s for NAs so they don't ruin the sums and means
df.fillna(0, inplace=True)
#Group by outgoing index, and aggregate
df = df.groupby(by=index_out).agg(agg)
df.head()
len(df)
#Just be extra sure there are no dupes
df = df[~df.index.duplicated(keep='first')]
#Just be extra sure no NaNs somehow slipped through
df.fillna(0, inplace=True)
df.describe()
#Write the output, with index field as one of the csv fields
#uncomment if you actually want to write

#df.to_csv(outfile, index=True)
df.head()
len(df)