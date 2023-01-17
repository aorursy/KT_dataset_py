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
%%time
donations_df = pd.read_csv('../input/Donations.csv')
%%time
donors_df = pd.read_csv('../input/Donors.csv', low_memory=False)
%%time
schools_df = pd.read_csv('../input/Schools.csv', error_bad_lines=False)
%%time
teachers_df = pd.read_csv('../input/Teachers.csv', error_bad_lines=False)
%%time
projects_df = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False, parse_dates=["Project Posted Date","Project Fully Funded Date"])
%%time
resources_df = pd.read_csv('../input/Resources.csv', error_bad_lines=False, warn_bad_lines=False)
%%time
import gc
gc.collect()
import numpy as np
import pandas as pd
from pathlib import Path

from IPython.display import display

def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000, 'display.width', 1000): 
        display(df)
    
    
def display_head(df):
    display(df.head())

    
def explore(df, sample_size=5, missing_value_columns=10):
    print("Info")
    display(df.info())
    
    print("Dataframe sample")
    display(df.sample(sample_size))

    print(f"{missing_value_columns} columns with maximum missing values")
    missing_values_count = df.isnull().sum().sort_values(ascending=False)

    display(missing_values_count[:missing_value_columns])
    
    display(df.describe())
    try:
        display(df.describe(include=['O']))
    except:
        print("No objects present")

    print("Percent of missing values")
    total_cells = np.product(df.shape)
    total_missing = missing_values_count.sum()

    print((total_missing / total_cells) * 100)
    
def to_string(df, column_name):
    df[column_name] = df[column_name].astype("|S")

def to_boolean(df, column_name, mapping):
    df[column_name] = df[column_name].map(mapping).astype(bool)
explore(donations_df)
to_string(donations_df, "Project ID")
to_string(donations_df, "Donation ID")
to_string(donations_df, "Donor ID")
to_boolean(donations_df, "Donation Included Optional Donation", {"Yes": True, "No": False})
donations_df.info()
import gc
gc.collect()