#Libraries Needed

import pandas as pd

import os

from glob import glob

from tqdm.notebook import tqdm



#Settings to display pandas

pd.set_option('display.max_columns', None)



#Some basic set up

base_path = "/kaggle/input/uncover"
#Traverse paths and pull out all .csv files

#Explaination at 

#https://perials.com/getting-csv-files-directory-subdirectories-using-python/

all_csv_files = []

for path, subdir, files in os.walk(base_path):

    for file in glob(os.path.join(path, "*.csv")):

        all_csv_files.append(file)

"There are a total of {} csv files in this dataset.".format(len(all_csv_files))
uncover_df_dictionary
#Try to make a pandas dataframe from each file

#The dataframes will be accessible by their file name using the uncover_df_dictionary

read_files = []

skipped_files = []

uncover_df_dictionary = {}

for file_path in tqdm(all_csv_files):

    df_name = file_path.split("/")[-1].replace('.csv','')

    try:

        uncover_df_dictionary[df_name] = pd.read_csv(file_path, low_memory=False)

        read_files.append(file_path)

    except:

        skipped_files.append(file_path)

        pass

"Read a total of {} files into Pandas dataframes and skipped {}.".format(len(read_files), len(skipped_files))
#Iterate though dict and group similar

column_dict = {}

for name, df in uncover_df_dictionary.items():

    all_cols = list(df.columns)

    for col in all_cols:

        if col in column_dict.keys():

            column_dict[col].append(name)

        else:

            column_dict[col] = list([name])



#Drop any columns not found in other DataFrames

len_before_drop = len(column_dict)

to_pop = []

for col, df_list in column_dict.items():

    if len(df_list) < 2:

        to_pop.append(col)

        

#Run in seperate loop as can not change size in iterator

for col in to_pop:

    column_dict.pop(col)

    

print("A total of {} columns are unique to one dataframe.".format((len_before_drop-len(column_dict))))

print("A total of {} columns are shared by more than one dataframe.".format(len(column_dict)))



#Make DF with index of cols and a column of dfs with that feature

col_df = pd.DataFrame(pd.Series(column_dict)).reset_index()

col_df.columns = ["Feature", "DataFrames"]

col_df.head(1)
#Explode-Make a new row for each of the values found in the DataFrames lists

col_df_explode = col_df.explode("DataFrames")

#Add present columns to keep track of which are where

col_df_explode["present"] = 1

#Pivot to binary matrix

col_binary_matrix = col_df_explode.pivot_table(index='Feature',

                    columns='DataFrames',

                    values='present',

                    aggfunc='sum',

                    fill_value=0)

col_binary_matrix.head()
col_binary_matrix.sum(axis=1).sort_values(ascending=False).head(10)
#Get all possible file combinaitons to compare

from itertools import permutations

all_pairs = permutations(col_binary_matrix.columns,2)

pairs_df_list = []

for df1, df2 in all_pairs:

    boolean_check = (col_binary_matrix[df1]==1) & (col_binary_matrix[df2]==1)

    shared_feats = list(col_binary_matrix.index[boolean_check])

    num_shared_feats = len(shared_feats)

    features_dict = {"df1":df1, "df2": df2,"sim_col_count": num_shared_feats,"sim_col_list": shared_feats}

    if num_shared_feats > 1:

        pairs_df_list.append(features_dict)

shared_cols_dfs = pd.DataFrame(pairs_df_list).sort_values("sim_col_count", ascending=False)

shared_cols_dfs.head(10)