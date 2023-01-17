import pandas as pd

import ast
# Variables

PATH_INPUT = "/kaggle/input/"

PATH_WORKING = "/kaggle/working/"

PATH_TMP = "/tmp/"
# Reading data into a df

df_raw = pd.read_csv(f'{PATH_INPUT}train.csv', low_memory=False, skipinitialspace=True)

df_raw.shape
# Take a look at the first 10 rows

df_raw.head(10)
# define columns with data of dict type to process

cols = ['belongs_to_collection', 'genres', 'production_companies', 'spoken_languages', 'Keywords', 'cast', 'crew']
# check the data type

df_raw[cols].dtypes
# copy the column to a pandas series

s = df_raw[cols[0]].copy()

s.shape
# check the first record

s[0]
# evaluate as a list

l = ast.literal_eval(s[0])

l
# check the data type

print(type(l), type(l[0]))
s[3]
# copy one column to a pandas series

s = df_raw[cols[0]].copy()

# fillna with [None]

s.fillna('[{}]', inplace=True)



l = []  # init an empty list



for i in s:

    if i == [{}]:

        # append [{}] to the list

        l += i

    else:

        # evaluate as a list

        l += ast.literal_eval(i)
len(l)  # should be 3000 if processed correctly
l[:10]
for i in range(10):

    print(type(l[i]))
df_tmp = pd.DataFrame.from_dict(l)

df_tmp[:10]
def to_list_of_dict(series):

    """

    Evaluate a pandas series as a list of dict

    

    Input:

    "[{'one': 1, 'two': 2, 'three': 3}]"

    

    Output:

    [{'one': 1,

      'two': 2,

      'three' : 3}]

    """

    l = []  # init an empty list

    s = series.fillna('[{}]')  # map nan to [{}] for further eval

    

    # loop through the whole series

    for i in s:

        if i == [{}]:

            # append [{}] to the list

            l += i

        else:

            # evaluate as a list

            l += ast.literal_eval(i)

    

    return l
def column_conversion(col, df):

    """

    Merge a pandas series with data like list of dict back to the dataframe

    

    Input:

    "[{'one': 1, 'two': 2, 'three': 3}]"

    

    Output:

    A dataframe with the original column removed, each dict's key in a new column

    """

    l = to_list_of_dict(df[col])  # convert to list of dict

    df_right = pd.DataFrame.from_dict(l)  # convert to df

    df_merged = df.merge(df_right.add_prefix(col+'_'),  # add the original column name as prefix

                         left_index=True, right_index=True)  # merge df with df_right

    df_merged.drop(col, axis=1, inplace=True) # drop the original column

    

    return df_merged
# Test

column_conversion(cols[0], df_raw)[:3]
# check the columns to process

cols
# make a copy

df = df_raw.copy()



# process the columns one by one

for col in cols:

    df = column_conversion(col, df)
# check the first record

df[:1]