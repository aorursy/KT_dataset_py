# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import hvplot.pandas
print(os.listdir("../input"))
from sympy import isprime, primerange

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/concorde-for-5-min/submission.csv')
cities = pd.read_csv("../input/traveling-santa-2018-prime-paths/cities.csv")
cities['isPrime'] = cities.CityId.apply(isprime)
prime_cities = cities.loc[cities.isPrime]
def score_path(path):
    cities = pd.read_csv('../input/traveling-santa-2018-prime-paths/cities.csv', index_col=['CityId'])
    pnums = [i for i in primerange(0, 197770)]
    path_df = cities.reindex(path).reset_index()
    
    path_df['step'] = np.sqrt((path_df.X - path_df.X.shift())**2 + 
                              (path_df.Y - path_df.Y.shift())**2)
    path_df['step_adj'] = np.where((path_df.index) % 10 != 0,
                                   path_df.step,
                                   path_df.step + 
                                   path_df.step*0.1*(~path_df.CityId.shift().isin(pnums)))
    return path_df
df2=score_path(df['Path'].values)
df2['Penalty']=df2['step_adj']-df2['step']
df3=df2[['CityId','Penalty']]
df4=df3[['Penalty']]
df4.hvplot()
sample = df2[['X','Y','Penalty']][:500]
sample.hvplot('X','Y') * sample.hvplot.scatter('X', 'Y',c='Penalty',size=15) 
#df4=df3.fillna(0).sort_values(by=['CityId'])
#df5=df4[['Penalty']].reset_index()[['Penalty']]
#df5.hvplot()
def penalty(df):
 return (df.values.reshape(197770)[np.arange(10,107770,10)]!=0).astype(int).sum()