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
df = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')

df.head() 
df.tail()
df.columns
df.shape
df.info()
print(df['Type 1'].value_counts(dropna=False))
df.describe()
df.boxplot(column='Attack',by='Legendary')
df_new = df.head()

df_new
melted = pd.melt(frame=df_new,id_vars='Name',value_vars=['Attack','Defense'])

melted
melted.pivot(index='Name',columns='variable',values='value')
df1 = df.head(3)

df2 = df.tail(3)

conc_df_row = pd.concat([df1,df2],axis =0,ignore_index =True)

conc_df_row
df1 = df.head(3)

df2 = df.tail(3)

conc_df_row = pd.concat([df1,df2],axis =1,ignore_index =True)

conc_df_row
df1 = df['Attack'].head()

df2 = df['Defense'].head()

conc_data_col = pd.concat([df1,df2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
df1 = df['Attack'].head()

df2 = df['Defense'].head()

df3 = df['Speed'].head()

conc_data_col = pd.concat([df1,df2,df3],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
df1 = df['Attack'].head()

df2 = df['Defense'].head()

conc_df_col = pd.concat([df1,df2],axis =1) # axis = 0 : adds dataframes in row

conc_df_col
df.dtypes
df['Type 1'] = df['Type 1'].astype('category')

df['Speed'] = df['Speed'].astype('float')
df.dtypes
df.info()
df["Type 2"].value_counts(dropna =False)
df1=df  

df1["Type 2"].dropna(inplace = True) 
assert 1==1 
#assert 1==2
assert  df['Type 2'].notnull().all()
df["Type 2"].fillna('empty',inplace = True)
assert  df['Type 2'].notnull().all()
assert df.Speed.dtypes == np.float