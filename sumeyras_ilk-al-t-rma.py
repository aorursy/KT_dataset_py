
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool(Görselleştirme Kütüphanesi)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/winequality-red.csv')
data.info()
data.columns
data.columns=[each.split()[0]+"_"+each.split()[1] if len(each.split())>1 else each for each in data.columns]    #feature lar boşluklu olduğu için; birleştirmek amaçlı
data.corr() #featurlar arasındaki korelasyon değerleri
f,ax=plt.subplots(figsize=(20,20))
sns.heatmap(data.corr(),annot=True, linewidth=.8,fmt='.2f',ax=ax)

plt.show()
data.count()
data.head(20)
#LINE PLOT

data.fixed_acidity.plot(kind="line", color='b' ,label="Fixed Acidity",linewidth=2 ,alpha=0.7,grid=True,linestyle="-" )
data.citric_acid.plot(color='r',label="Citric Acid", linewidth=1, alpha=1 ,grid=True, linestyle=":")

plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()



#SCATTER PLOT

data.plot(kind='scatter', x='free_sulfur', y='total_sulfur', alpha=0.2, color='red')

plt.xlabel("Free Sulfur")
plt.ylabel("Total Sulfur")
plt.title("Scatter Plot")
plt.legend()
plt.show()


#HISTOGRAM
data.alcohol.plot(kind="hist",bins=50,figsize=(23,23))
plt.show()
print('pH' in data)
series=data['alcohol']   # Vektör
print(type(series))

dataFrame=data[['alcohol']]
print(type(dataFrame))
quality_filter=data.quality>7
data[quality_filter]
data[(data.quality>7) & (data.citric_acid>0.5)]
quality_filter=data.quality>7
for index,value in enumerate(data[quality_filter].fixed_acidity):
    print(index,":",value)
    
print('')

for index,value in data[['alcohol']][0:2].iterrows():
    print(index,":",value)
