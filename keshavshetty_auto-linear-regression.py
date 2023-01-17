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
# Install Auto Linear Regression (Part of KUtils Package)

!pip install kesh-utils
# Load the custom packages from kesh-utils

from KUtils.eda import chartil

from KUtils.eda import data_preparation as dp
# The library is auto_linear_regression we give alias as autolr

from KUtils.linear_regression import auto_linear_regression as autolr
# Use 3 decimal places for decimal number (to avoid displaying as exponential format)

pd.options.display.float_format = '{:,.3f}'.format
import warnings  

warnings.filterwarnings('ignore')
# Load the dataset

diamond_df = pd.read_csv('../input/diamonds.csv')
# Have a quick look on the top few records of the dataset 

diamond_df.head()
diamond_df.describe()
# Drop first column which is just a sequence

diamond_df = diamond_df.drop(diamond_df.columns[0], axis=1)
diamond_df.shape
# Null checks

diamond_df.isnull().sum()
# Plot the nulls as barchart (Null count in each features)

dp.plotNullInColumns(diamond_df)
# Number of unique values in each column (Check in both Train and Test for missing categorial label in any)

{x: len(diamond_df[x].unique()) for x in diamond_df.columns}
# Plot unique values in each feature

dp.plotUnique(diamond_df, optional_settings={'sort_by_value':True})
# Some EDA (Using kesh-utils chartil package)

chartil.plot(diamond_df, diamond_df.columns, optional_settings={'include_categorical':True, 'sort_by_column':'price'})
# Have a quick look on different feature and their relation

chartil.plot(diamond_df, ['carat', 'depth','table', 'x','y','z','price', 'cut'], chart_type='pairplot', optional_settings={'group_by_last_column':True})
chartil.plot(diamond_df, ['color', 'price'], chart_type='distplot')
chartil.plot(diamond_df, ['clarity', 'price'])
model_info = autolr.fit(diamond_df, 'price', 

                     scale_numerical=True, acceptable_r2_change = 0.005,

                     include_target_column_from_scaling=True, 

                     dummies_creation_drop_column_preference='dropFirst',

                     random_state_to_use=100, include_data_in_return=True, verbose=True)
# Print all iteration info

model_info['model_iteration_info'].head()
# Final set of features used in final model 

model_info['features_in_final_model']
# vif-values of features used in final model  

model_info['vif-values']
# p-values of features used in final model  

model_info['p-values']
# Complete stat summary of the OLS model

model_info['model_summary']