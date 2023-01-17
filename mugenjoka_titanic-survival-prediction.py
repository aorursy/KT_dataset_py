# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv', header=0)
print(df.head(2))

print(df.dtypes)

print(df.info())

print(df.describe())
# Before jumping in, lets understand the data and describe it using graphs and statistics, which needs to be executed 2 times 

###Before Cleaning

###After Ceaning

### Do cleaning of data first



# Cleaning the data, items to resolve

### 1. Pclass, Survived, Sex into values

### 2. Age: solve for null values 

### 3. ticket: Into meaning values

### 4. Cabin: Into Meaningful values ( <25% of the data set however may be meaningfulto increase accuracy later on)

### 5. Embarked: Solve for 2 missing values, and make into a categorical

### 6. Sibs, Parch: Into categorical based on presumed information

### 7. Passenger ID: check for duplicates/ drop

### 8. Name: Break into First Name/Last Name, study for family names and add to data set to use later
def df_value_counts(col_list):

    df_vc_dict={}

    for each in df[col_list]: ### May need to check for null

        df_vc_dict[each]=df[each].value_counts()

    return df_value_counts
# Creating list of column names 

print(df.columns.values)

col_names=np.array(df.columns.values)



# Checking for Passenger ID Count
