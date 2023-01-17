# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plot
import seaborn as sns #visualization
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Pokemon.csv")
data.info()
data.corr()
f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head()
data.tail()
data.columns
data.shape
data.describe
print(data['Type 2'].value_counts(dropna =False))
data.describe()
data.boxplot(column='Speed',by = 'Legendary')
data_new1 = data.tail()
data_new1
melted = pd.melt(frame=data_new1,id_vars = 'Name', value_vars= ['Speed','Legendary'])
melted
data_1 = data.head()
data_2 = data.tail()
con_data = pd.concat([data_1,data_2],axis=0) #for vertical axis =0
con_data
data1 = data['Defense'].head()
data2 = data['Speed'].head()
con_data_2= pd.concat([data1,data2],axis=1) # for horizontal axis=1
con_data_2
data.dtypes
data['Type 1'] = data['Type 1'].astype('category')
data['Speed'] = data['Speed'].astype('float') # for chance type
data.dtypes #yeni type list 
data1=data   
data1["Type 2"].dropna(inplace = True)
assert  data['Type 2'].notnull().all()
data["Type 2"].fillna('empty',inplace = True)
assert  data['Type 2'].notnull().all()
