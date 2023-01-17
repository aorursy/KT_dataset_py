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
#verimimiz import ediyoruz 
data = pd.read_csv('../input/pokemon.csv')

#veri hakkında bilgi alırız örneğin türünü sütünlarda boşluk varmı ? 
data.info();
#özellikler arası korelasyonu bulabiliriz . 
data.corr()
#coorelations map  (özellikler arasındaki ilişkiyi anlamamızı sağlayan belli başlı parametreler )

f,ax = plt.subplots(figsize=(18, 18))
#data corr şekilli grafikli yazmaya çalışıyor 
#(annot = true (0,1 , 0,4 gibi yazıların görünür olmasına yarıyor )  linewidths = linenın kalınlığı , fmt ='.1f' sıfırdan sonra bir degeği yazdır diyoruz , ax figürün size'1  )
sns.heatmap(data.corr() , annot=True , linewidths=.5  , fmt='.1f', ax = ax   )
plt.show();
#ilk  10 pokemonu sıralar 
data.head(10)
#column sahip olduğu özellikleri göstereiyor 
data.columns
data.Speed.plot(kind ='line' , color='g' , label = 'Speed' , linewidth = 1 , alpha = 0.5 , grid =True , linestyle ='-.')
data.Defense.plot (color ='r' , label = 'Defensse ', linewidth = 1,alpha=0.5 , grid=True , linestyle = '-.')
plt.legend(loc ='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Speed vs Defense ')