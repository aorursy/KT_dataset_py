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
###object Creation -series,dataframe



series=pd.Series([1,'mani',3.14,np.nan])

#print(series)

dates=pd.date_range('2019-07-15',periods=7) 

#print(dates) 

dataframe=pd.DataFrame(np.random.randn(7,8),index=dates,columns=list('ABCDEFGH')) 

#print(dataframe)

dict_df=pd.DataFrame({'A':1.,

                      'B':pd.Timestamp('20190715'),

                      'C':pd.Series(1,index=list(range(4)),dtype='float32'),

                      'D':pd.Categorical(["test","train","test","train"]), 

                      'E':np.array([4]*4,dtype='int32'),

                      'F': 'foo'

                      }) 

#print(dict_df) 

#dict_df.F or dict_df['F']

dict_df.dtypes

dict_df.columns 

dataframe.index 
###viewing data  

head=dataframe.head()   

#print(head)

tail=dataframe.tail(3) 

#print(tail)

desc=dataframe.describe() 

#print(desc)  

#dataframe.to_numpy()  

#print(num)

#num1=dict_df.to_numpy() 

#print(num1)   

print(dataframe)

print(dataframe.T)

print(dataframe.sort_index(axis=0,ascending=False))  

print(dataframe.sort_values(by='B'))
dataframe['A'] 

dict_df.loc[dict_df['A']] 

dict_df.groupby('D').sum() 

dict_df.groupby(['F','D']).sum() 

dict_df.isna() 

dataframe.fillna(value=5) 

dataframe.dropna(how='any') 

#dict_df.value_counts() 

dict_df.str.lower()
####selection 



#print(dataframe['A'])   

#print(dataframe[0:3])  

####Selection by label

print(dataframe['2019-07-17':'2019-07-19'])

print(dataframe.loc[dates[0]]) 

print(dataframe.loc[:,['A','B']])

print(dataframe.loc['2019-07-15':'2019-07-16',['A','B']]) 

print(dataframe.loc['2019-07-15',['A','B']]) 

print(dataframe.loc[dates[0],'A']) 

print(dataframe.at[dates[0],'A']) 





#####Selection by Position 

dataframe.iloc[3] 

dataframe.iloc[3:5,2:4] 

dataframe.iloc[[1,4],[2,3]] 

dataframe.iloc[[2,3],:] 

dataframe.iloc[:,[3,5]] 

dataframe.iloc[4,5]
#######Boolean indexing 

dataframe[dataframe.A>0]  

dataframe[dataframe > 0]  

df=dataframe.copy() 


