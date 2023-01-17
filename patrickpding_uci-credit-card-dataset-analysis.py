# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', 1000)

pd.set_option('display.width', 1000)

pd.set_option('display.max_colwidth', 1000)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Import basic libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv('../input/UCI_Credit_Card.csv')

#print(df.sample(5))

print(df.info())

#get the name of features/entries

print(df.columns)
# Extracting # of unique entires per column and their sample values

num_unique = []

sample_col_values = []

for col in df.columns:

    num_unique.append(len(df[col].unique()))  # Counting number of unique values per each column

    sample_col_values.append(df[col].unique()[:5])  # taking 3 sample values from each column



# combining the sample values into a a=single string (commas-seperated)

# ex)  from ['hi', 'hello', 'bye']  to   'hi, hello, bye'

col_combined_entries = []

for col_entries in sample_col_values:

    entry_string = ""

    for entry in col_entries:

        entry_string = entry_string + str(entry) + ', '

    col_combined_entries.append(entry_string[:-2])

# Generating a list 'param_nature' that distinguishes features and targets

param_nature = []

for col in df.columns:

    if col == 'default.payment.next.month':

        param_nature.append('Target')

    else:

        param_nature.append('Feature')



# Generating Table1. Parameters Overview

df_feature_overview = pd.DataFrame(np.transpose([param_nature, num_unique, col_combined_entries]), index = df.columns, columns = ['Parameter Nature', '# of Unique Entries', 'Sample Entries (First three values)'])

print(df_feature_overview)
print(pd.value_counts(df["MARRIAGE"]))

print(pd.value_counts(df["PAY_0"]))

print(pd.value_counts(df["EDUCATION"]))
#分割 data target

X=df[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',

       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',

       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',

       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].copy()

y=df['default.payment.next.month'].copy()

print(X.head())

print(y.head())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train,y_train)

train_score=model.score(X_train,y_train)

test_score=model.score(X_test,y_test)

print('train score:{train_score:.6f}; test score:{test_score:.6f}'.format(train_score=train_score,test_score=test_score))
