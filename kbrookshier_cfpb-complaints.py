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
import pandas as pd

complaints = pd.read_csv("../input/complaints.csv")
# # Row display; None for print all

# pd.set_option('display.max_rows', 100)
# Print out the top 5 rows of the dataframe

complaints.head()
# Number of rows and columns in the df

complaints.shape
# Print out a list of all the column names in the df

complaints.columns
# Print the number of columns in the df

len(complaints.columns)
# Copy the dataframe with only a few of the columns

# df1 = df[['a','b']]

complaints_small = complaints.sample(frac=0.05)[['Product','Issue','Company public response','Company','State']]

complaints_small.head()
complaints_small.groupby('State').size().sort_values(ascending=False)
# df.loc[df['column_name'] == some_value]

complaints_small.loc[complaints_small['State'] == 'WA']
# Group types of products, sort descending

complaints_small.groupby('Product').size().sort_values(ascending=False)
# Group types of issues, sort descending

complaints_small.groupby('Issue').size().sort_values(ascending=False)
# Group companies, sort descending

complaints_small.groupby('Company').size().sort_values(ascending=False).head(100)
# Group companies, sort descending

complaints_small.groupby('Company public response').size().sort_values(ascending=False)
complaints_small.groupby(["Product","Issue"]).size().head(100)