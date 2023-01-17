import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns  

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import os
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')
data.head()

data.info()
data.corr()

#1 veya 1'e yakın çıkarsa doğru orantılı bir korelasyon var
#0'a yakın çıkarsa iki feature birbiri ile alakasız (speed ve generation gibi)
#correlation map

f,ax=plt.subplots(figsize=(13,13)) #figsize'i önceden belirlemek istedik 

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

#koyu renkler korelasyonu en düşük olanları ifade etmektedir.

plt.show()
data.head(10)
data.columns
data.Speed.plot(kind='line',color='g',label='Speed',linewidth=1,alpha=0.5,grid=True,linestyle=':')
data.Defense.plot(color='r',label='Defense',linewidth=1,alpha=0.5,grid=True,linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()

data.plot(kind="scatter", x="Attack", y="Defense",alpha=0.5,color="red")
plt.xlabel("Attack")
plt.ylabel("Defense")
plt.title("Attack Defense Scatter Plot")
plt.show()
plt.scatter(data.Attack,data.Defense,color="red",alpha=0.5)
plt.show()
