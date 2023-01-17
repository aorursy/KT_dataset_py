# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np

import pandas as pd

import itertools

from sklearn import preprocessing
df = pd.read_csv("../input/loan.csv", low_memory=False)
def duplicate_columns(df, return_dataframe = False, verbose = True):

    '''

        a function to detect and possibly remove duplicated columns for a pandas dataframe

    '''

    from pandas.core.common import array_equivalent

    # group columns by dtypes, only the columns of the same dtypes can be duplicate of each other

    groups = df.columns.to_series().groupby(df.dtypes).groups

    duplicated_columns = []



    for dtype, col_names in groups.items():

        column_values = df[col_names]

        num_columns = len(col_names)

 # find duplicated columns by checking pairs of columns, store first column name if duplicate exist 

        for i in range(num_columns):

            column_i = column_values.iloc[:,i].values

            for j in range(i + 1, num_columns):

                column_j = column_values.iloc[:,j].values

                if array_equivalent(column_i, column_j):

                    if verbose: 

                        print("column {} is a duplicate of column {}".format(col_names[i], col_names[j]))

                    duplicated_columns.append(col_names[i])

                    break

    if not return_dataframe:

        # return the column names of those duplicated exists

        return duplicated_columns

    else:

        # return a dataframe with duplicated columns dropped 

        return df.drop(labels = duplicated_columns, axis = 1)
df.columns
df.shape
df['loan_status'].value_counts(normalize=True)
df = df.loc[df['loan_status']!="Current"]

df.shape
df = duplicate_columns(df, return_dataframe = True)
df.shape
df.head()
vc = df.member_id.value_counts()

print("# members: ", len(vc[vc>0]))

print("# reoccuring members: ", len(vc[vc>1]))
df.drop(["id",'url'],axis=1,inplace=True)
df['loan_Default'] = int(0)

for index, value in df.loan_status.iteritems():

    if value == 'Default':

        df.set_value(index,'loan_Default',int(1))

    if value == 'Charged Off':

        df.set_value(index, 'loan_Default',int(1))

    if value == 'Late (31-120 days)':

        df.set_value(index, 'loan_Default',int(1))    

    if value == 'Late (16-30 days)':

        df.set_value(index, 'loan_Default',int(1))

    if value == 'Does not meet the credit policy. Status:Charged Off':

        df.set_value(index, 'loan_Default',int(1))    
df['loan_Default'] .describe()
# Drop original label column

df.drop(["loan_status"],axis=1,inplace=True)
df.sample(5000).to_csv("LC_defaultLoansK_5k.csv.gz",index=False,compression="gzip")
df.sample(250000).to_csv("LC_defaultLoansK_250k.csv.gz",index=False,compression="gzip")