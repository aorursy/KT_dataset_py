# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/globalterrorismdb_0718dist.csv', engine='python')
dataset
dataset.shape
dataset.head()

dataset.info()
dataset.tail(5)
dataset.columns
dataset.describe()
dataset.country_txt.unique()
country =dataset[dataset.country_txt =="Turkey"]
country.describe()
dataset.head(3)

country.targtype1.plot(kind ='line',color ='g',label='month',alpha= 0.5,grid =True,linestyle=':')
country.targsubtype1.plot(color='r',label='year',alpha =0.5,grid=True,linestyle = '-.')
plt.legend()
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()


#scatter

country.plot(kind ='scatter', x ='targtype1', y ='targtype2',alpha = 0.5,color = 'red' )

plt.title('Years and Month Scatter Plot')

#histogram
plt.hist(country.imonth,bins=20)

#series
series =dataset['targtype1']
type(series)

df =dataset[['targtype1']]
df
x = df['targtype1']<12
df[x]
dataset[np.logical_and(dataset['targtype1'], dataset['targtype2'])]
dataset[np.logical_and(dataset['iyear']>1970, dataset['imonth']>10)]
data_new =dataset.head()
data_new
#melt
melted =pd.melt(frame = data_new,id_vars='region_txt', value_vars =['attacktype1','iyear'])
melted
#pivot
melted.pivot(index = 'region_txt', columns ='variable', values ='value')