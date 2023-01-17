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
path="../input/INDUSTRIAL_EXPORTERS DATABASE.xls"

df=pd.read_excel(path)

df.head()
new_header = df.iloc[1]

df = df[3:] 

df.columns = new_header
df.head()
df=df[df.columns.dropna()]
df.head()
df.describe(include='all')
df['COMPANY NAME'].value_counts()
df['PRODUCT'].value_counts()
#df.filter(like='DELHI', axis=1)

df.reset_index()

df2=df.copy()

df2
df2=df.drop(df.columns[2],axis=1)

df2
df=df.reset_index()
df=df.drop(['index'],axis=1)
df
df.index.name=None
df.head()
DelhiStatus = df[df['ADDRESS'].str.contains('DELHI', na=False)]

DelhiStatus.head()
TechStatus = df[df['COMPANY NAME'].str.contains('ENGINEER', na=False)]

TechStatus.head()
TechStatus['MOBILE NO'].head(50)