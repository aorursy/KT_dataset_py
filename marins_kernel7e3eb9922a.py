# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

def import_data():
    # Import data as dataframe, and tidy up column names to Python friendly
    
    # COVID-19 cases
    df_cov = pd.read_csv("../input/covid19-novel-coronavirus-croatia-cases/covid_hr.csv", low_memory=False)
    df_cov.columns = df_cov.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    # Drop the city column since no detail provided on population
    df_cov = df_cov.drop('city', 1)
    
    # Republic of Croata population stats, also remove 2018_ prefix
    df_pop = pd.read_csv("../input/croatia-bureau-of-statistics-2018-population/DE_SP22_2.csv", low_memory=False)
    df_pop.columns = df_pop.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    df_pop.columns = df_pop.columns.str.lstrip('2018_')
    return df_cov, df_pop

df_cov, df_pop = import_data()

df_cov.dtypes

# df_cov.cumsum()
# df_cov
# df_pop
