# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/heart-disease-uci/heart.csv")
data.head() #ilk 5'ini gösterir
data.tail() #son 5'ini gösterir
data.info()
data.corr() #correlation birbirleri arasındaki oran,doğru yada tesrs orantı gibi
import seaborn as sns
import matplotlib.pyplot as plt
#kind=türü ,marker=düz çizgi yerine hangi şekilde çizilecekse,alpha=saydamlık,linestle=çizgi stili,linewitdth=çizgi genişliği
data.trestbps.plot(kind="line",color="red",marker="*",label="trestbps",linewidth=3,alpha=0.5,linestyle=":")
data.age.plot(kind="line",color="g",linewidth=2,linestyle="-")
plt.legend=("trestbps","age")
plt.x_label=("x axis")
plt.title=("line plot")
data.plot(kind="scatter",x="age",y="trestbps",color="r")
plt.show() #üstteki siyah olan grafiğin bilgisini kaldırır.

data.age.plot(kind="hist",color="g",alpha=0.75,bins=80,histtype="step",orientation="vertical",rwidth=0.2) #bins=kutu sayısı,histtype=histogram tipini belirler,
#rwidth=çucukların genişliğinin belirler

#dictionary
sözlük={"isim":"ali","yaş":"41"} #dictionary oluşturuldu
sözlük["iş"]="veribilimci" #dictionary'e yeni birşey eklemek
sözlük
sözlük["isim"]="veli" #isim değiştirildi
sözlük
del sözlük["iş"] #sözlükten iş silindi
sözlük
"yaş" in sözlük #sözlükte yaş varmı yokmu?
#del sözlük #sözlüğü tamamen sile(hafızadan silinir)
sözlük.clear() #sözlüğün içindekiler silinir sözlük tamemen silinmez
sözlük
#pandas
seri=data["age"]
type(seri)

dataframe=data[["age"]] #dataFrame oluşturmak için 2 tane liste ile yapılmalıdır
type(dataframe)
age30=data.age<30 #age'de 30 küçük değerleri
age30.head()  #bu değerlerin ilk 5 verisi
data[np.logical_and(data["age"]<50,data["chol"]>300)] #data'nın içindeki age'nin 50 'den küçük chol'un 300 den büyük değerlerini andlıyor her ikiside True ise veriler gösteriliyor
(data["age"]<50)&(data["chol"]>300) #yukardakinin aynısı farklı bir gösterimi ile yanlışlarda gösteriliyor 
#while-for
i=2
while i<10:  #2 den başlayarak 9 a kaar yazdıran döngü
    print(i)
    i+=1
    

array=np.array([1,2,4,5,7,88])
for i in array:  #arrayin içindeki değerleri 1 arttıran döngü
    i+=1
    print(i)
liste=[4,5,6,7,8]
for index,value in enumerate(liste):  #enumerate ile hangi indexte hangi value deperi var gösterilir
    print(index,":",value)
    
sözlük={"isim":"ali","soyisim":"veli"}
for key,value in sözlük.items(): #diztionary'de key ve valur değerleri .items ile bulunur
    print(key,":",value)
    
