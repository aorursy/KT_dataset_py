# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Install Clustered Linear Regression (Part of KUtils Package)

!pip install kesh-utils
# Load the custom packages from kesh-utils

from KUtils.eda import chartil

from KUtils.eda import data_preparation as dp

from KUtils.linear_regression import auto_linear_regression as autolr

from KUtils.linear_regression import clustered_linear_regression as clustlr
# Some warning from pandas and Numpy need to ignore for time being (Some like conversion from int to float, cyclyic subset etc)

import warnings  

warnings.filterwarnings('ignore')
# Use 3 decimal places for decimal number (to avoid displaying as exponential format)

pd.options.display.float_format = '{:,.3f}'.format
# Load the dataset

diamond_df = pd.read_csv('../input/diamonds.csv')
# Have a quick look on the top few records of the dataset 

diamond_df.head()
diamond_df.describe()
# Drop first column which is just a sequence

diamond_df = diamond_df.drop(diamond_df.columns[0], axis=1)
diamond_df['price'] = diamond_df['price'].astype(float) # One of the warning can be escaped
diamond_df.head()
# Auto Linear Regression - Single model for entire dataset

model_info = autolr.fit(diamond_df, 'price', 

                     scale_numerical=True, acceptable_r2_change = 0.005,

                     include_target_column_from_scaling=True, 

                     dummies_creation_drop_column_preference='dropMin',

                     random_state_to_use=44, include_data_in_return=True, verbose=True)
# model_iteration_info

model_info['model_iteration_info'].head()
group_model_info, group_model_summary = clustlr.fit(diamond_df, feature_group_list=['cut', 'color','clarity'], dependent_column='price', 

                                                    max_level = 1, min_leaf_in_filtered_dataset=500,

                                                    verbose=True)
# Check the modle summary

group_model_summary
# Check subgroup level model efficieny and the dataset size used for each subset

group_model_info
# Do some visualization and analysis on feature 'clarity' and see why it performs better when it is split using that feature

chartil.plot(diamond_df, ['clarity', 'price'], chart_type='violinplot')