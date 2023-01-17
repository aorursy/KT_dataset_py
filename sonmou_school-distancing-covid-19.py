#Import Packages



import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt





from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity='all'

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#/kaggle/input/CORD-19-research-challenge/Kaggle/target_tables/2_relevant_factors/Effectiveness of school distancing.csv



#read file

fname = '/kaggle/input/CORD-19-research-challenge/Kaggle/target_tables/2_relevant_factors/Effectiveness of school distancing.csv'

school_dist_df = pd.read_csv(fname)

school_dist_df
school_dist_df.shape
school_dist_df.info()
school_dist_df.head(10)

school_dist_df.tail(10)
school_dist_df.columns
school_dist_df.groupby('Journal')['Unnamed: 0'].count()
school_dist_df.groupby('Date')['Unnamed: 0'].count()
school_dist_df.groupby('Added on')['Unnamed: 0'].count()
school_dist_df.groupby('Measure of Evidence')['Unnamed: 0'].count()
school_dist_df.groupby('Influential')['Unnamed: 0'].count()
school_dist_df.groupby('Study Type')['Unnamed: 0'].count()








journal_moe_df1 = school_dist_df.groupby(['Journal','Measure of Evidence'])['Unnamed: 0'].aggregate('count')

journal_moe_df1

journal_moe_df1.index



journal_moe_df1 = school_dist_df.groupby(['Journal','Measure of Evidence'])['Unnamed: 0'].aggregate('count').unstack()




def summary_table(col1, col2, col_index, data_frame):

    var1 = col1 + '_' + col2 + '_df1' 

    print(var1)

    var1 = data_frame.groupby([col1,col2])[col_index].aggregate('count')

    print(var1)

    print('\n')

    print(var1.index)

    print('\n')

    

    var2 = col1 + '_' + col2 + '_df2'

    print(var2)

    var2 = data_frame.groupby([col1,col2])[col_index].aggregate('count').unstack()

    print(var2)

    print('\n')

    var2



    var3 = col1 + '_' + col2 + '_df3'

    print(var3)

    var3 = var2.fillna(0)

    print(var3)

    print('\n')





    arr = var3.values

    print('arr')

    print(arr)

    print('\n')

    

    # Create a new row at bottom to hold COLUMN TOTALS

    new_row = np.zeros(shape=(1,arr.shape[1]))

    new_row



    # Vertically stack new row of zeroes at bottom

    arr1 = np.vstack([arr, new_row])

    print('arr1')

    print(arr1)

    print('\n')



    # Create a new column at right to hold ROW TOTALS

    new_col = np.zeros(shape=(arr1.shape[0],1))

    new_col



    #arr1.shape

    #new_col.shape





    # Horizontally stack new column of zeroes at right

    arr2 = np.hstack([arr1, new_col])

    print('arr2')

    print(arr2)

    print('\n')

    

    arr3 = arr2.copy()

    



    # Fill last row with sum of all values in each column

    arr3[ arr3.shape[0]-1 ]=arr2.sum(axis=0) # column total

    

    



    # Fill last column with sum of all values in each row

    arr3[ :, arr3.shape[1]-1 ] = arr3.sum(axis=1) # row total

    print('arr3')

    print(arr3)

    

summary_table('Journal', 'Date', 'Unnamed: 0', school_dist_df)


journal_moe_df1 = school_dist_df.groupby(['Journal','Measure of Evidence'])['Unnamed: 0'].aggregate('count')

journal_moe_df1

journal_moe_df1.index



journal_moe_df1 = school_dist_df.groupby(['Journal','Measure of Evidence'])['Unnamed: 0'].aggregate('count').unstack()