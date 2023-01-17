import gc

import os

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')



import json

import subprocess

from pandas.io.json import json_normalize



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



pd.options.display.max_columns = 999

pd.options.display.max_rows = 999
PATH="../input/"

os.listdir(PATH)
json_path = PATH + 'text.data.jsonl.xz'
import lzma



def sample(json_path, sample_size=0.25, num_lines=None, drop=[]):

    """

    Sample json file

    """

        

    def file_len(fname):

        """

        Given filename, return number of lines

        Credits: Ólafur Waage, https://stackoverflow.com/a/845069/3096104

        """

        p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 

                                                  stderr=subprocess.PIPE)

        result, err = p.communicate()

        if p.returncode != 0:

            raise IOError(err)

        return int(result.strip().split()[0])

    

    sample_size = sample_size * file_len(json_path)

    

    with lzma.open(json_path) as in_file:

        lines = 0

        sample = []

        line = in_file.readline()

        if num_lines:

            while (line and lines < num_lines):

                case = json.loads(str(line, 'utf8'))

                        

                # drop elements 

                for to_drop in drop:

                    temp = case

                    for obj in to_drop.split('.')[:-1]:

                        if obj in temp:

                            temp = temp[obj]

                    if to_drop.split('.')[-1] in temp:

                        del temp[to_drop.split('.')[-1]]

                                

                sample.append(case)

                lines = lines + 1

                line = in_file.readline()

        else:

            while (line and lines < sample_size):

                case = json.loads(str(line, 'utf8'))

                # drop elements 

                for to_drop in drop:

                    temp = case

                    for obj in to_drop.split('.')[:-1]:

                        if obj in temp:

                            temp = temp[obj]

                    if to_drop.split('.')[-1] in temp:

                        del temp[to_drop.split('.')[-1]]

                        

                sample.append(case)

                lines = lines + 1

                line = in_file.readline()

        return sample
illinois_sample = sample(json_path, sample_size=0.01) # sample the data
illinois_sample = json_normalize(illinois_sample)

illinois_sample.head(2)

illinois_sample.shape
def expand(df):

    """

    Expand list and dict columns of Pandas dataframe.

    

    Given a df where certain columns contains list of JSON object or python dictionary, return expanded df.

    

    Columns with NAN values are not expanded.

    """

    

    # expand list

    for col in df:

        # check if all columns contain dictionary df

        is_list = df[col].apply(lambda x: isinstance(x, list)).any()

        has_na = df[col].isna().any()

        

        # if it's a list, expand the df, merge and delete legacy column

        if is_list and not has_na:

            temp = df[col].apply(pd.Series)

            temp.columns = temp.columns.astype(str)

            temp.columns = col + '_' + temp.columns

            df = df.drop(columns=[col]).join(temp, lsuffix='.' + col )

            

    # expand dict

    for col in df:

        # check if all columns contain dictionary df

        is_dict = df[col].apply(lambda x: isinstance(x, dict)).any()

        has_na = df[col].isna().any()

        

        # if it's a dict, expand the df, merge and delete legacy column

        if is_dict and not has_na:       

            normalized = json_normalize(df[col])

            normalized.columns = col + '.' + normalized.columns

            df = df.drop(columns=[col]).join(normalized, how='outer')

    return df



illinois_sample = expand(illinois_sample)

illinois_sample.head(1)
def lowercase(df):

    """Return a df with all object columns lowercased"""

    object_columns = df.select_dtypes('object')

    for col in object_columns:

        df[col] = df[col].str.lower()

    return df



illinois_sample = lowercase(illinois_sample)

illinois_sample.head(1)
illinois_sample = illinois_sample.loc[:,~illinois_sample.isna().all()]

illinois_sample.head(1)
def is_list(series):

    """Return true if all columns of pd.Series are list"""

    return series.apply(lambda x: isinstance(x, list)).any()



def is_dict(series):

    """Return true if all columns of pd.Series are dict"""

    return series.apply(lambda x: isinstance(x, dict)).any()



def columns_list_dict(df):

    """Return all columns of df that are either list or dict"""

    columns_list_dict = []

    for col in df:

        series = df[col]

        if is_list(series) or is_dict(series):

            columns_list_dict.append(col)

    return columns_list_dict



unhashable_columns = columns_list_dict(illinois_sample)

unhashable_columns
illinois_sample, hashable_data = illinois_sample.drop(columns=unhashable_columns), illinois_sample[unhashable_columns]
illinois_sample.nunique()
print("'id' is unique?:", illinois_sample['id'].is_unique)
illinois_sample = illinois_sample.loc[ :, (illinois_sample.nunique() != 1) ]
illinois_sample.head(1)

illinois_sample.loc[0, 'casebody.data.head_matter']
illinois_whole = sample(json_path, sample_size=1, drop=['casebody.data.head_matter', 'casebody.data.opinions']) 



illinois_whole = json_normalize(illinois_whole)



illinois_whole.head(2)
illinois_whole = expand(illinois_whole)
illinois_whole = lowercase(illinois_whole)
illinois_whole = illinois_whole.loc[:,~illinois_whole.isna().all()]
illinois_whole = illinois_whole.loc[ :, (illinois_whole.nunique() != 1) ]
illinois_whole.head(1)



print("There are ", illinois_whole.shape[0], "law cases")
illinois_whole['court.name'].value_counts()
illinois_whole[['court.name', 'court.name_abbreviation']].drop_duplicates()
illinois_whole = illinois_whole.drop(columns=['court.name_abbreviation'])
def plot_nan(df):

    from matplotlib import cm



    some_nan = df.isnull().sum() != 0



    nan_percentage = df.isna().sum() / df.shape[0]

    nan_percentage.sort_values(inplace=True)



    cmap = cm.get_cmap('coolwarm')

    colors = cmap(nan_percentage)



    nan_percentage.plot(kind='bar', figsize=(20,8), title="Percentage of NaN", color=colors, fontsize=14)



plot_nan(illinois_whole)
attorney_columns = [col for col in illinois_whole.columns.tolist() if 'attorneys' in col]

illinois_whole['all_attorneys'] = illinois_whole[attorney_columns].apply(lambda x: x.dropna().tolist(), axis=1)

illinois_whole = illinois_whole.drop(columns=attorney_columns)
judges_columns = [col for col in illinois_whole.columns.tolist() if 'judges' in col]

illinois_whole['all_judges'] = illinois_whole[judges_columns].apply(lambda x: x.dropna().tolist(), axis=1)

illinois_whole = illinois_whole.drop(columns=judges_columns)
plot_nan(illinois_whole)
assigned_parties = illinois_whole.loc[:,(illinois_whole.isna().sum() > 0)].dropna()



assigned_parties
illinois_whole.head(10)