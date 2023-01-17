# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print (check_output(["ls","../input"]).decode("ISO-8859-1"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
data.info()
data.corr()
data.head(5)
f,ax = plt.subplots(figsize=(100,100))
sns.heatmap(data.corr(),annot=True,linewidths=.5, fmt='.2f',ax=ax)
data.head(10)
data.columns
data.plot(kind='scatter', x='iyear',y='attacktype1',color='r', alpha=1)

plt.xlabel('iyear')
plt.ylabel('attacktype1 ')
plt.title('Line Plot')
data.attacktype1.plot(kind='hist', bins=50, figsize=(15,15))
plt.show()
plt.clf()
series=data['iyear']
print(type(series))
data_frame=data[['iyear']]
print(type(data_frame))
x=data['iyear']>2016
data[x]
data[np.logical_and(data['iyear']>2016,data['attacktype1']>8)]
for index,value in  data[['iyear']][0:10].iterrows():
    print(index,':',value)