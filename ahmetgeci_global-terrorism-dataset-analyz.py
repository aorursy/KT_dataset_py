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

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/globalterrorismdb_0718dist.csv' ,encoding='ISO-8859-1' )
data.info()
data.corr()
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot =True,fmt = '.1f', linewidths =.5 ,ax=ax )
data.head()
plt.scatter(data.country , data.latitude,color='red')
ulkebul = list(data.country_txt)
ulkebul.count("Turkey")
#Turkiyede yapılan teror saldıralrının sayısını bulmak istiyoruz 
#örnek bir teror saldırısını bulalım mesela
# ulkebul.index("Turkey") dediğimizde 226 sonucunu vericektir 
#buda 226 ıncı indexte turkiyeede yapılan saldırı hakkında bilgi verir(havalimanı pat)


print(data["country_txt"].value_counts(dropna=False))
data.country.plot(kind='hist' , bins = 50 , color = 'blue' ,)