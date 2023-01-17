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
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df
#df1 = df["Age"]



df.iloc[0:9,1:3]
df
df3 = df[df.Age>30]







df_X=df3.drop("Survived",axis=1)



df_X.dtypes



df_int=pd.DataFrame()



for i in df_X:

    cnt=0

    if df_X[i].dtypes!='object':

        

        df_int=pd.concat([df_int,df_X[i]],axis=1)

        cnt=cnt+1

        

        

df_int

        

lst=[100,233,'10',33,34,5330]



for i in lst:

    if type(i)=="<class 'int'>":

        print(i,'is a string')

    

        

    
lst=[100,233,10,33,34,5330]



df_lst = pd.DataFrame(lst)



df_lst.dtypes