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
df = pd.read_csv('../input/expenditure-by-salisbury-nhs-foundation-trust/expenditure.csv')

df.head()
df.isna().sum()
# Drop all VAT registration numbers since they are all missing



df.drop(["VAT registration Number", "VAT Registration Number", "VAT registration number"], axis=1, inplace=True)
print ("Adding the number of NaN values in the two Amount columns gives:", 748572 + 56227,". The total number of rows is {}".format(len(df)))

print ("The number of NaN values in the Expense column is:", 804799 - 795092, ".")
# Rename columns for clarity 



df.columns = ["Department Family", "Entity", "Date", "Expense Type 1", "Expense Area", "Supplier", "Transaction Number", "Amount 1", "Amount 2", "Expense Type 2"]
df.head()
df["Amount 1"].fillna(df["Amount 2"], inplace=True)

df["Expense Type 1"].fillna(df["Expense Type 2"], inplace=True)



# Check all Nan filled



print (df["Amount 1"].isna().sum())

print (df["Expense Type 1"].isna().sum())
# Drop unnecessary columns



df.drop(["Amount 2", "Expense Type 2"], axis=1, inplace=True)
df.head()
df.rename(columns={"Expense Type 1": "Expense Type", "Amount 1": "Expenditure"}, inplace=True)

df.head()
df.to_csv('expenditure_v2.csv',index=False)