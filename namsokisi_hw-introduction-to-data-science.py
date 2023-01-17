# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Iris.csv')
#Information Data (Veri ile ilgili bilgi)
data.info()
#Correlation (Korelasyon yani ilişkileri gösterir)
data.corr()
#Correlation Map (Korelasyon Haritasını Gösterir)
f,ax = plt.subplots(figsize=(8,5)) 
sns.heatmap(data.corr(), annot=True, linewidths=.2, fmt= '.3f',ax=ax)
plt.show()

#Line Plot (Çizgi Grafiği)
data.SepalLengthCm.plot(kind = 'line', color = 'blue',label = 'Sepal Length',linewidth=1,alpha = 1.7,grid = True,linestyle = ':')
data.PetalLengthCm.plot(color='black',label = 'Petal Length',linewidth=1, alpha = 1.7,grid = True,linestyle = '-.')
plt.legend(loc='lower right')
plt.xlabel("x axis(x ekseni)",color="purple")
plt.ylabel("y axis(y ekseni)",color="green")
plt.title('Line Plot (Çizgi Grafiği)',color="yellow")

#Scatter Plot (Saçılım Grafiği)
data.plot(kind="scatter", x="SepalWidthCm", y="PetalWidthCm",alpha = 0.99,color = "blue")
plt.xlabel("Sepal Width",color="orange")             
plt.ylabel("Petal Width",color="green")
plt.title("Sepal and Petal Scatter Plot (Sepal ve Petal'in Saçılım Grafiği)",color="yellow")  
#Histogram
data.SepalWidthCm.plot(kind="hist",bins=30,color="Orange")
plt.title("Sepal Width Histogram",color="green")

import pandas as pd
data=pd.read_csv("../input/Iris.csv")
#Filter Pandas  data frame (pandas da data frame filtreleme)
data[(data["SepalLengthCm"]>6) & (data["PetalLengthCm"]>5.5)] 

i = 1
while i !=12 :
    print("i is:",i)
    i += 1
print(i,"is equal to 12")


xyz = [1,2,3,4,5,6,7,8,9,10]
for i in xyz:
    print("i is:",i)
print("")    
data = pd.read_csv("../input/Iris.csv")
data.head() #First 5 rows (ilk 5 satır)
data.tail() #Last 5 rows (son 5 satır)
data.columns #columns gives column namesof features (sütun isimlerini verir)
data.shape #shape gives number of rows and columns in a tuble (kaç satır kaç sütun olduğunu gösterir)
data.info() #info gives type like dataframe, number of sample or row, number of feature or column, feature types etc. (bize dataframeler, satır sütundaki bilgileri verir )
print(data["Species"].value_counts(dropna=False))
#value_counts **  data frequency list (verideki sıklık listesini verir)**
data.describe()
#shows us count,std,min,median etc. (bize toplam,meydan,ortalama,çeyrekleri gösterir)
# Black line at top is max ( üstteki siyah çizgi en büyük değer)
# Blue line at top is 75% (üstteki mavi çizgi 3.Çeyrek)
# Red line is median (50%) (Kırmızı çizgi medyan(ortanca))
# Blue line at bottom is 25% (alttaki mavi çizgi 1.çeyrek)
# Black line at bottom is min (alttaki siyah çizgi en küçük değer)
data.boxplot(column="SepalLengthCm",by = "Species")
data_new =data.head()  
data_new
#Firstly I creating new dataset from Iris data to explain melt nore easily.(öncelikle kolayca melt yapmak için yeni veriseti oluşturuyorum)
melted = pd.melt(frame=data_new,id_vars = 'Species', value_vars= ["PetalWidthCm","PetalLengthCm"])
melted
#lets melt (hadi melt yapalım)
#id_vars =what we dont wish to melt (melt yapmak istemediğimiz yer)
#value_vars= what we want to melt (melt yapmak istediğimiz yer)
data.dtypes
#shows us data types (bize değişkenlerin veri tiplerini gösterir)
#lets convert object (object) to categorical and int to float.(bject veri tipini kategorik yapalım)
data["Species"] = data["Species"].astype("category")

data.dtypes
#yes we did it! (evet yaptık!)
data.info() 
#we have not nAn data so we can pass this. (boş gözlemimiz yok bu yüzden burayı geçebiliriz)
