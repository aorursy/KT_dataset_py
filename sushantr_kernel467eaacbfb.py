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
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
print("# of Rows & Columns = ", df.shape)
df.head()
for col in list(df.columns):
    print(col, " :: Unique values = ", len(df[col].unique()))
for col in ["Patient age quantile", "SARS-Cov-2 exam result", "Inf A H1N1 2009"]:
    print(col, " :")
    df[col].value_counts()

df["SARS-Cov-2 exam result"].value_counts()
df["Patient age quantile"].value_counts()
df["Inf A H1N1 2009"].value_counts()
list(df.columns[~df.isna().any()]) #Provide a list of columns with 0 missing values.
for col in list(df.columns[~df.isna().any()]):
    print("Columns = ", col, " :: Unique values = ", len(df[col].unique()))
