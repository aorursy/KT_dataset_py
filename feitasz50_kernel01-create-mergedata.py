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
df = pd.read_csv('../input/igiristoast_utf8.csv')

df.head()
df.shape
result=df['name'].str.replace('(','（')

result = result.str.replace(')','）')

result = result.str.replace('）','')

result = result.str.split('（')
flavor = []

for i in range(len(result)):

  flavor.append(result[i][1])



columns=[["flavor"]]

flavor = pd.DataFrame(data=flavor, columns=columns)

flavor.head()
df['flavor']=flavor

df.head()
flavor_df=pd.read_csv('../input/flavor_lut_utf8.csv')

flavor_df.head(10)

fcolumns = flavor_df['groupname'].unique()

fparts_df = pd.DataFrame(data = np.zeros((len(df), len(fcolumns)),dtype=np.int),columns=fcolumns) 

fparts_df.head()
#各行のflavor列について

for i in range(len(df)):

#flavor_dfの各行の用語が含まれているかチェック

  for fidx in range(len(flavor_df)):

    if flavor_df.iloc[fidx,0] in df.iloc[i,-1] :

      fparts_df[flavor_df.iloc[fidx,1]][i]=1

      

#含まれていればfparts_dfの対象のカラムのフラグを立てる
fparts_df.head()
merge_df = pd.concat([df, fparts_df], axis=1)

merge_df.head()
#merge_df.to_csv('merge_igiris.csv',index=False)