# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/world-happiness/2019.csv") 
df.head()
df.info() # Datamız hakkında bilgi sahibi oluyoruz
df.columns # Datamızdakı hangi satırlar var onlara bakıyoruz
df.corr() # Verimizin değerleri arasındaki korelasyona bakıyoruz
sns.heatmap(df.corr(), annot=True,) # Bu korelasyonu grafik halinde inceliyoruz
df.Score.plot(kind = 'line', color = 'red',label = 'Score',linewidth=1,grid = True,linestyle = ':') # Ülkelerin sıralamasına göre mutluluk seviyelerine bakıyoruz

plt.legend(loc='upper right')

plt.xlabel('Ülke Sıralaması')

plt.ylabel('Mutluluk Seviyesi')              

plt.title('Ülkelere Göre Mutluluk Seviyesi') 

plt.show()
df.plot(kind='scatter', x='Score', y='Social support',color = 'blue') # Ülkelerin Mutluluk seviyesi ve sosyal destek arasındaki ilişkiye bakıyoruz 

plt.xlabel('Ülke Skoru')              

plt.ylabel('Sosyal Destek')

plt.title('Mutluluk Seviye ve Sosyal Destek Arasındaki İlişki')   

plt.show()
df.plot(kind='scatter', x='Score', y='Perceptions of corruption',color = 'red') # Mutluluk seviyesi ve yolsuzluk algısı arasındaki ilişkiye bakıyoruz

plt.xlabel('Score')              # label = name of label

plt.ylabel('Perceptions of corruption')

plt.title('Mutluluk Seviyesi ve Yolsuzluk Algısı Arasındaki İlişki')              

plt.show()
df['Healthy life expectancy'].plot(kind = 'hist',bins = 50,figsize = (10,10))

plt.show()