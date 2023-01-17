# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import



import pandas as pd

import numpy as np

import os



import networkx as nx



from tqdm import tqdm_notebook

import tqdm



import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

%matplotlib inline



import seaborn as sns



pd.options.display.float_format = '{:,.1f}'.format
#utility



def Insert_row(row_number, df, row_value): 

    # Slice the upper half of the dataframe 

    df1 = df[0:row_number].copy()

   

    # Store the result of lower half of the dataframe 

    df2 = df[row_number:].copy()

   

    # Inser the row in the upper half dataframe 

    df1.loc[row_number]=row_value 

   

    # Concat the two dataframes 

    df_result = pd.concat([df1, df2]) 

   

    # Reassign the index labels 

    df_result.index = [*range(df_result.shape[0])] 

   

    # Return the updated dataframe 

    return df_result 
# Load the initial data

data_dir = "/kaggle/input/prozorro-public-procurement-dataset/"

data_suppliers = "Suppliers.csv"



# # Check all data files

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



df_supl = pd.read_csv(os.path.join(data_dir, data_suppliers), index_col=0, dtype="str")

df_supl[["lot_initial_value", "lot_final_value"]] = df_supl[["lot_initial_value", "lot_final_value"]].astype(float)

df_supl.index = pd.to_datetime(df_supl.index)



print(f"The shape of the DF: {df_supl.shape[0]:,.0f} rows, {df_supl.shape[1]:,.0f} columns")

display(df_supl.head(5).T)
year = df_supl['lot_announce_year'].unique()

list_ave_init = []

list_ave_final = []

for y in year:

    list_ave_init.append(df_supl[df_supl['lot_announce_year']==y]['lot_initial_value'].mean())

    list_ave_final.append(df_supl[df_supl['lot_announce_year']==y]['lot_final_value'].mean())

    

plt.plot(year, list_ave_init, label='lot_initial_value', marker='o' )

plt.plot(year, list_ave_final, label='lot_final_value', marker='o' )

plt.legend()

plt.ylabel('Average of value')
arr_save_value = np.array(list_ave_init) - np.array(list_ave_final)

plt.bar(year, arr_save_value, label='Diff init-final')

plt.legend()
#Data of 99_Other in 2015 is missing

df_ave_values = df_supl.groupby(['lot_cpv_2_digs', 'lot_announce_year']).mean().reset_index()

df_ave_values = Insert_row(225, df_ave_values,['99_Other', '2015', 0.0, 0.0])



df_ini_value = pd.DataFrame([])

df_fin_value = pd.DataFrame([])

for col in df_ave_values['lot_cpv_2_digs'].unique():

    tmp = df_ave_values[df_ave_values['lot_cpv_2_digs']==col][['lot_announce_year', 'lot_initial_value']]

    df_add = pd.DataFrame(tmp['lot_initial_value'].values, columns=[col], index=tmp['lot_announce_year'])

    df_ini_value = pd.concat([df_ini_value, df_add], axis=1)

    

    tmp = df_ave_values[df_ave_values['lot_cpv_2_digs']==col][['lot_announce_year', 'lot_final_value']]

    df_add = pd.DataFrame(tmp['lot_final_value'].values, columns=[col], index=tmp['lot_announce_year'])

    df_fin_value = pd.concat([df_fin_value, df_add], axis=1)

    
for n,cpv in enumerate(df_ini_value.columns):

    

    if n%3 == 0:

        plt.figure(figsize=(20,5))

        plt.subplots_adjust(wspace=0.8)

    

    ini_v = df_ini_value[cpv]

    fin_v = df_fin_value[cpv]

    

    plt.subplot(1,3,n%3+1)

    plt.plot(year,ini_v.T.values, label='lot_initial_value')

    plt.plot(year,fin_v.T.values, label='lot_initial_value')

    plt.ylabel('Average of value')

    plt.legend()

    plt.title(''.join(cpv.split('_')[1])+'\n', fontdict={'fontsize':10})
df_diff_ave = df_ini_value - df_fin_value



plt.figure(figsize=(15,13))

for cpv in df_diff_ave.columns:

    plt.plot(year, df_diff_ave[cpv], label=''.join(cpv.split('_')[1]))

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
for n, y in enumerate(year):

    print('Diff(initial - final) Top 3 in %s' % y)

    print(df_diff_ave.sort_values(y,ascending=False,axis=1).iloc[n,:3])

    print('\n')
