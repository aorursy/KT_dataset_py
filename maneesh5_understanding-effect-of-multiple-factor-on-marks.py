# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
# Any results you write to the current directory are saved as output.
# read the data
df = pd.read_csv("../input/StudentsPerformance.csv")
# initial understanding of the daya
df.head()
df.describe()
df.dtypes
# some fields are objects converting them to categorical variable
cat_cols = df.dtypes.pipe(lambda x: x[x == 'object']).index
num_cols = df.dtypes.pipe(lambda x: x[x == 'int64']).index
for c in cat_cols:
    df[c] = df[c].astype('category')
df.dtypes
display(pd.crosstab(index=df["gender"], columns="count"))
display(pd.crosstab(index=df["race/ethnicity"], columns="count"))
display(pd.crosstab(index=df["lunch"], columns="count"))
display(pd.crosstab(index=df["test preparation course"], columns="count"))
display(pd.crosstab(index=df["parental level of education"], columns="count"))

df.describe()
plt.hist([df['writing score'], df['reading score'], df['math score']], label=['writing', 'reading', 'math'])
plt.legend(loc='upper right')
plt.show()
df['avg'] = (df['writing score']+ df['reading score']+df['math score'])/3
for cat_col in cat_cols:
    display(df.groupby([cat_col]).describe())

for cat_col in cat_cols:
    df.boxplot(column=['writing score','reading score','math score','avg'], by=cat_col)


