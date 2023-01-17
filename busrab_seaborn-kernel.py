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
import seaborn as sns
import matplotlib.pyplot as plt
Iris=pd.read_csv("../input/Iris.csv", encoding="windows-1252")
Iris.head()
dataFrame=Iris.loc[:,['Id','SepalLengthCm','Species']]
dataFrame['Species'].unique()
dataFrame.info()

a_list=list(dataFrame['Species'].unique())
a_SepalLengthCm=[]
for i in a_list:
    x=dataFrame[dataFrame['Species']==i]
    area_SepalLengthCmSum=sum(x.SepalLengthCm)/len(x)
    a_SepalLengthCm.append(area_SepalLengthCmSum)
data=pd.DataFrame({'a_list':a_list,'a_SepalLengthCm':a_SepalLengthCm})
new_index=(data['a_SepalLengthCm'].sort_values(ascending=False)).index.values
sorted_data=data.reindex(new_index)

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 5), sharex=True)

sns.barplot(x=sorted_data['a_list'], y=sorted_data['a_SepalLengthCm'], palette="rocket", ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
ax1.set_ylabel("Sequential")
a_list=list(dataFrame['Species'].unique())
a_SepalLengthCm=[]
for i in a_list:
    x=dataFrame[dataFrame['Species']==i]
    area_SepalLengthCmSum=sum(x.SepalLengthCm)/len(x)
    a_SepalLengthCm.append(area_SepalLengthCmSum)
data=pd.DataFrame({'a_list':a_list,'a_SepalLengthCm':a_SepalLengthCm})
new_index=(data['a_SepalLengthCm'].sort_values(ascending=False)).index.values
sorted_data=data.reindex(new_index)
plt.figure(figsize=(15,10))
#seaborn kütüphanesi:sns
sns.barplot(x=sorted_data['a_list'], y=sorted_data['a_SepalLengthCm'])
#x eksenindeki yazıları 45 derecelik açıyla koyduk
plt.xticks(rotation= 45)
plt.xlabel('States')
plt.ylabel('Poverty Rate')
plt.title('Poverty Rate Given States')
