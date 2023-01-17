#Imports
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set(style="whitegrid")

import os
print(os.listdir("../input"))
def get_null_cnt(df):
    """Return pandas Series of null count encounteres in DataFrame, where index will represent df.columns"""
    null_cnt_series = df.isnull().sum()
    null_cnt_series.name = 'Null_Counts'
    return null_cnt_series

def plot_ann_barh(series, xlim=None, title=None, size=(12,6)):
    """Return axes for a barh chart from pandas Series"""
    #required imports
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    #setup default values when necessary
    if xlim == None: xlim=series.max()
    if title == None: 
        if series.name == None: title='Title is required'
        else: title=series.name
    
    #create barchart
    ax = series.plot(kind='barh', title=title, xlim=(0,xlim), figsize=size, grid=False)
    sns.despine(left=True)
    
    #add annotations
    for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(i.get_width()+(xlim*0.01), i.get_y()+.38, \
                str(i.get_width()), fontsize=10,
    color='dimgrey')
    
    #the invert will order the data as it is in the provided pandas Series
    plt.gca().invert_yaxis()
    
    return ax
pdb = pd.read_csv('../input/pdb_data_no_dups.csv', index_col='structureId')
pdb_seq = pd.read_csv('../input/pdb_data_seq.csv', index_col='structureId', na_values=['NaN',''], keep_default_na = False)
pdb.head()
# See what's the highest number of sequence that a single protein has
pdb_seq_cnt = pdb_seq['chainId'].groupby('structureId').count()
print('The MAX number of sequences for a single protein is {}'.format(pdb_seq_cnt.max()))
print('The AVERAGE number of sequences for a single protein is {}'.format(pdb_seq_cnt.mean()))
print('The pdb dataset has {} rows and {} columns'.format(pdb.shape[0], pdb.shape[1]))
print('The pdb_seq dataset has {} rows and {} columns'.format(pdb_seq.shape[0], pdb_seq.shape[1]))
msno.matrix(pdb.sample(1000))
# sanity check on missingno
# build null count series
pdb_null_cnt = get_null_cnt(pdb)
# plot series result
ax = plot_ann_barh(pdb_null_cnt, xlim=len(pdb), title='Count of Null values for each columns in PDB DataFrame')
msno.matrix(pdb_seq.sample(1000))
# sanity check on missingno
# build null count series
pdb_seq_null_cnt = get_null_cnt(pdb_seq)
# plot series result
ax = plot_ann_barh(pdb_seq_null_cnt, xlim=len(pdb_seq), title='Count of Null Values in PDB Sequence DataFrame', size=(12,2))
# get index for lines with null values in chainId column
null_chain_idx = list(pdb_seq[pdb_seq['chainId'].isnull()].index)

# get count per structureId which has at least one line where chainId column is null
null_chain_cnt = pdb_seq.loc[null_chain_idx,'chainId'].groupby('structureId').count()


print(list(null_chain_idx))
print(null_chain_cnt[null_chain_cnt > 1])
# get index for lines with null values in sequence column
null_seq_idx = list(pdb_seq[pdb_seq['sequence'].isnull()].index)

# get count per structureId which has at least one line where sequence column is null
null_seq_cnt = pdb_seq.loc[null_seq_idx,'chainId'].groupby('structureId').count()

print(null_seq_idx)
print(null_seq_cnt[null_seq_cnt > 1])
print('Class of Protein Structure without sequence:')
class_no_seq = pdb.loc[null_seq_idx,'classification'].value_counts()
class_no_seq
print('Total count (across dataset) of classes with at least 1 protein structure without sequence:')
class_no_seq_total = pdb[pdb['classification'].isin( list( class_no_seq.index))]['classification'].value_counts()
class_no_seq_total
# create Dataframe with the 2 Series we created above
class_df = pd.concat([class_no_seq, class_no_seq_total], axis=1)
class_df.columns=['no_seq_cnt', 'total_cnt']

# Create a new ratio column to understand how many of the total count don't have sequence
class_df['ratio'] = class_df.apply(lambda row: round(100* (row['no_seq_cnt'] / row['total_cnt']), 2) ,axis=1)

# Sort values and print
class_df = class_df.sort_values('ratio', ascending=False)
class_df
print('The number of unique values in each of PDB dataset categorical columns')
print(pdb.select_dtypes(include=object).nunique())
expTech_counts = pdb['experimentalTechnique'].value_counts(dropna=False)
ax = plot_ann_barh(expTech_counts, title='CountPlot for experimentalTechnique', size=(12,8))
macroType_counts = pdb['macromoleculeType'].value_counts(dropna=False)
ax = plot_ann_barh(macroType_counts, title='CountPlot for macromoleculeType', size=(12,4))
#since this feature has over 500 values, let's have look at the top20 one only
crystMethod_counts = pdb['crystallizationMethod'].value_counts(dropna=False)[:20]
ax = plot_ann_barh(crystMethod_counts, title='CountPlot for Top20 crystallizationMethod values')
pdb.select_dtypes(exclude=object).describe().T
#let's look at the publication year a bit closer before we look at incorrect values
pdb_pubYear = pdb['publicationYear']
print('publicationYear has {} unique values, from {} to {}.'.format(pdb_pubYear.nunique(), int(pdb_pubYear.min()), int(pdb_pubYear.max())))

# Now for the number of instance we have with incorrect years
pdb_pubYear_str = pdb_pubYear.fillna(0).astype('int').astype('str')
# 0 was added as possible value in the regex pattern since it's the value I used for filling in missing values in that column
pattern = '(0|(19|20)\d{2})'
correct_year = pdb_pubYear_str.str.match(pattern, na=False)
print('\nThe incorrect year values and their respective counts:')
pdb_pubYear_str[~correct_year].value_counts()
# get class and then get all records in protein db with the same class
bad_year_class = pdb[~correct_year]['classification'].tolist()
bad_year_class_pubYear = pdb[pdb['classification'].isin(bad_year_class)]['publicationYear']

print('The mean publicationYear for class {} is {}'.format(bad_year_class[0], 
                                                           int(bad_year_class_pubYear.mean())
                                                          )
     )

print('The most used publicationYear for class {} is {}'.format(bad_year_class[0], 
                                                                #the next line is not pretty - but it gets the first value from
                                                                # value_counts series, since the series is always ordered by value_count
                                                                int(list(bad_year_class_pubYear.value_counts().index)[0])
                                                               )
     )

pdb.loc[~correct_year, 'publicationYear'] = 2011

# sanity check
pdb[~correct_year]
print('There are {} classes in this dataset'.format(pdb['classification'].nunique()))
classes_counts = pdb['classification'].value_counts(dropna=False)[:50]
ax = plot_ann_barh(classes_counts, title='CountPlot for Top50 Classes', size=(12,10))






