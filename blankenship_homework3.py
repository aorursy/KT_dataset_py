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
data = pd.read_csv('../input/dc-wikia-data.csv')
data.head()
data.tail()
data.columns
data.shape
data.info()
filter1 = data[(data['SEX'] == 'Male Characters')&(data['ID']=='Secret Identity')&(data['ALIGN']=='Good Characters')]

filter1
filter1.shape
filter1.describe()
data['ALIVE'].value_counts(dropna = False)
filter1[filter1['name']=='Batman (Bruce Wayne)']
filter2 = data[(data['ALIVE']=='Living Characters')&(data['APPEARANCES']>1200)]

filter2
filter2.describe()
filter2.boxplot(column ='APPEARANCES')
melted_data = pd.melt(frame = filter2,id_vars = 'name',value_vars = ['FIRST APPEARANCE','APPEARANCES'])

melted_data
melted_data.pivot(index = 'name',columns = 'variable',values = 'value') #I think, We don't need melted data.
filter2
data1 = filter2['EYE']

data2 = filter2['HAIR']

data_concat = pd.concat([filter2['name'],data1,data2],axis = 1)

data_concat
filter2.dtypes
filter2.info()
filter2['GSM'].value_counts(dropna = False)
fd = filter2.copy()

fd
fd['GSM'].dropna(inplace = True)
assert fd['GSM'].notnull().all()
fd
fd['GSM'].fillna('empty',inplace = True)  
fd.notnull().all()
fd