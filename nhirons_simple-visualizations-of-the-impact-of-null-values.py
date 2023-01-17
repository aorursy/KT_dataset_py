# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
# Replace -1 values as nulls

df = df.replace(-1,np.NaN)



# Create isnull dataframe

df_nan = df.isnull()



# Create df only including columns with null values

df_nan = df_nan[[col for col in df_nan.columns if df_nan[col].sum() > 0]]



# Examine number of unique values

{c: len(df[c].unique()) for c in df_nan.columns}
# Create dataframe of the ratio of target 1's for null / non-null values in each column



# Create empty list to contain series of the proportion of target 1's for null and non-null

nan_vs_true_list = []



# For each column

for col in df_nan.columns:

    

    # Cross tabulate target 1's and 0's vs. null and non-null values for each column

    nan_vs_target = pd.crosstab(df['target'], df_nan[col])

    

    # Convert to ratio

    nan_vs_target = nan_vs_target / nan_vs_target.sum()

    

    # Consider only the target 1's

    nan_vs_true = nan_vs_target.loc[1,:]

    

    # Append this series to list

    nan_vs_true_list += [nan_vs_true]



# Create dataframe of the ratio of target 1's for null / non-null values in each column

df_nan_vs_true = pd.concat(nan_vs_true_list, axis = 1, keys = df_nan.columns)

df_nan_vs_true
# Visualize the proportion of target 1's for null vs. non-null values

melted = pd.melt(df_nan_vs_true.T.reset_index(), id_vars = 'index')



f,axarray = plt.subplots(1,1,figsize=(15,8))



sns.barplot(data = melted, x = 'index', y = 'value', hue = 'variable', palette = 'husl')



plt.title('Proportion of target truths for null vs. non-null values')

plt.xticks(rotation = 'vertical')



plt.xlabel('Feature')

plt.ylabel('Proportion of target true values')

plt.legend(title = 'Null value?')



plt.show()
# Examine the ratio of target 1's for null vs. non-null values

df_nan_vs_true.loc['Null ratio'] = df_nan_vs_true.loc[True] / df_nan_vs_true.loc[False]

df_nan_vs_true
f,axarray = plt.subplots(1,1,figsize=(15,8))

plt.title('Ratio of target truths for null vs. non-null values')

plt.xticks(rotation = 'vertical')

bins = df_nan_vs_true.loc['Null ratio']

sns.barplot(x=bins.values,y=bins.index,orient='h')

plt.show()