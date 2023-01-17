import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
s = pd.Series([1,2,3,4,5])

s
s = pd.Series([1,3,5,np.nan,6,8])

s
#Creating a DataFrame by passing a numpy array, with a datetime index and labeled columns



dates = pd.date_range('20130101', periods=9)

dates

df = pd.DataFrame(np.random.rand(9,4) , index=dates, columns=list('ABCD'))

df
df2 = pd.DataFrame({ 'A' : 1.,

                     'B' : pd.Timestamp('20130102'),

                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),

                     'D' : np.array([3] * 4,dtype='int32'),

                     'E' : pd.Categorical(["test","train","test","train"]),

                     'F' : 'foo' })

df2
#Creating a DataFrame by passing a dict of objects that can be converted to series-like.



df2 = pd.DataFrame({ 'Date': pd.Timestamp('20190311'),

                     'Next Date': pd.Timestamp('20190312'),

                     'Daily Spending Money': pd.Series(10,index=list(range(5)),dtype='float32'),

                     'Person': np.array([1] * 5,dtype='int32')

                     #'Possition' : pd.Categorical(["Intern","parmanent","Intern","Intern","Ceo"])

                     #'Name' : pd.name(["Mr S Roy","Dr Kumar","Tommy","Imma","Tojo"]) 

                   })

df['M'] = ['one', 'one','two','three','four','three']

