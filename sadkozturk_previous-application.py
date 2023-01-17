# numpy and pandas for data manipulation

import numpy as np

import pandas as pd 



# sklearn preprocessing for dealing with categorical variables

from sklearn.preprocessing import LabelEncoder



# File system manangement

import os



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')



# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

import seaborn as sns
# List files available

print(os.listdir("../input/home-credit"))
# previous_application data

previous_application = pd.read_csv('../input/home-credit/previous_application.csv')

print('previous_application data shape: ', previous_application.shape)

previous_application.head()
# Function to calculate missing values by column# Funct 

def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
missing_values_table(previous_application)
previous_application['NAME_CONTRACT_TYPE'].value_counts()
previous_application['AMT_ANNUITY'].value_counts()
previous_application['AMT_ANNUITY'].describe()
plt.hist(previous_application['AMT_ANNUITY'])
plt.hist(previous_application['AMT_APPLICATION']-previous_application['AMT_CREDIT'])
plt.hist(previous_application['WEEKDAY_APPR_PROCESS_START'])

previous_application['WEEKDAY_APPR_PROCESS_START'].value_counts()
plt.hist(previous_application['HOUR_APPR_PROCESS_START'])

previous_application['HOUR_APPR_PROCESS_START'].value_counts()
application = pd.read_csv('../input/home-credit/application_train.csv')

previous_application = pd.read_csv('../input/home-credit/previous_application.csv')
PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = []

for agg in ['mean', 'min', 'max', 'sum', 'var']:

    for select in ['AMT_ANNUITY',

                   'AMT_APPLICATION',

                   'AMT_CREDIT',

                   'AMT_DOWN_PAYMENT',

                   'AMT_GOODS_PRICE',

                   'CNT_PAYMENT',

                   'DAYS_DECISION',

                   'HOUR_APPR_PROCESS_START',

                   'RATE_DOWN_PAYMENT'

                   ]:

        PREVIOUS_APPLICATION_AGGREGATION_RECIPIES.append((select, agg))

PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], PREVIOUS_APPLICATION_AGGREGATION_RECIPIES)]


from tqdm import tqdm_notebook as tqdm

groupby_aggregate_names = []

for groupby_cols, specs in tqdm(PREVIOUS_APPLICATION_AGGREGATION_RECIPIES):

    group_object = previous_application.groupby(groupby_cols)

    for select, agg in tqdm(specs):

        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)

        application = application.merge(group_object[select]

                              .agg(agg)

                              .reset_index()

                              .rename(index=str,

                                      columns={select: groupby_aggregate_name})

                              [groupby_cols + [groupby_aggregate_name]],

                              on=groupby_cols,

                              how='left')

        groupby_aggregate_names.append(groupby_aggregate_name)
application.head()
application_agg = application[groupby_aggregate_names + ['TARGET']]

application_agg_corr = abs(application_agg.corr())
application_agg_corr.sort_values('TARGET', ascending=False)['TARGET']