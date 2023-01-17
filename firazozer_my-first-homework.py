# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #visualition part..

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Pandas kütüphanesinden yararlanarak CSV uzantılı dosyayı import ederiz.

data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv") #import etme

data.head(10) # bu metodla verilerin ilk 10 tanesini görmemize imkan sağlar default değer olsaydı 5 değer gözükürdü.
data.info() # Datasetinde bulunan verilerin özellikleri hakkında bilgi verir. Datanın boyunu vb.
data.corr() # Datalardaki featurelarının arasındaki ilişkiyi gösteren metod(correlation method)
f,ax = plt.subplots(figsize =(15,15))

sns.heatmap(data.corr(), annot = True, linewidth = .5, fmt = '.1f') 

# correlation tableda kimi zaman verileri görmek zor olduğu için seaborn kütüphanesi ile verileri görselleştirerek daha anlaşılır hala getirmek için uygulanan metod

#bu yönteme corralation map yöntemi deni
data.columns
data.age.plot(kind ='line',color = 'r', label = 'age',linewidth = 1,alpha = 0.5, grid = True, linestyle = '-')

data.thalach.plot(kind ='line',color = 'g', label = 'Max Hearth Rate',linewidth = 1,alpha = 0.5, grid = True, linestyle = '-')

plt.legend(loc = 'upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line plot') #Grafiğe isim vermek için. Daha sonra değiştirilicek.

plt.show()
data.plot(kind = 'scatter', x='age', y='chol', alpha=0.5, color='r')

plt.xlabel('Fastin Blood Sugar')

plt.ylabel('Cholestoral')

plt.title('Fasting Blood Sugar and Cholestoral Scatter Plot')
data.age.plot(kind = 'hist', bins = 50)

#filtering data with PANDAS

x = data['age']>55 #burada yaşı 50den büyük olan insanları filter etmesini istedik ve x'e atadık

data[x] #data değişkeninin içine x yazarak koşulu sağlayan dataları döndürecek. eğer herhangi bir değer yazmasaydık true ve false olarak döndürecekti
#filterin pandas with logical_and

data[np.logical_and(data['age']>55 , data['chol']>350)] # burada da iki koşulu sağlarsa tablola dedim.





#logical and kullanmadan and operatörüyle filtreleme 



data [(data['age'] > 55) & (data['chol']>350)]
#Data setin içerisinden bazı datalara erişim sağlamak veya featureslara göre filtreme veya sıralama yapmak için kullanılan for loop yöntemi.

for index,value in data[['age']][0:5].iterrows():

    print (index, " : ", value)