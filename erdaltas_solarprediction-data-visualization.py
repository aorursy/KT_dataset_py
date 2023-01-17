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
veri=pd.read_csv("../input/SolarPrediction.csv") 
veri.head() #ilk 5 örnek
veri.info() #parametreler ve veri tipleri
temp=list(veri.Temperature.unique()) 
radiation=[]
basinc_degerleri=[]#bu değişken ortalama sıcaklık-basınç grafiği içindir.
for i in temp:
    x=veri[veri["Temperature"]==i]
    average_radiation=sum(x.Radiation)/len(x)
    radiation.append(average_radiation)
    #2.bilgi: sıcaklık-basınç grafiği için değişkenler
    ortalama_basinc=sum(x.Pressure)/len(x)
    basinc_degerleri.append(ortalama_basinc) 
data=pd.DataFrame({"Temperature":temp,"Radiation":radiation})
new_index=(data["Radiation"].sort_values(ascending=True)).index.values
sorted_data=data.reindex(new_index)    
sorted_data.head() 
#sıcaklık-radrasyon bar plot
plt.figure(figsize=(10,10))
sns.barplot(x=sorted_data.Temperature,y=sorted_data.Radiation,data=data)
plt.xticks(rotation=45)
plt.xlabel("Temperature")
plt.ylabel("Average Radiation")
plt.title("Temperature-Average Radiation comparasion",fontsize=20)
plt.show() 
#heatmap
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(veri.corr(),annot=True,ax=ax) 
#2.bilgi: sıcaklık-basınç grafiği
data2=pd.DataFrame({"Sıcaklık":temp,"Basinc":basinc_degerleri})
new_index2=(data2["Basinc"].sort_values(ascending=True)).index.values
sorted_data2=data2.reindex(new_index2)
sorted_data2.head() 
#sıcaklık-basınç bar plot 
plt.figure(figsize=(10,10))
sns.barplot(x="Sıcaklık",y="Basinc",data=data2)
plt.xlabel("Sıcaklık")
plt.xticks(rotation=45)
plt.ylabel("Basınç")
plt.title("Sıcaklık-Basınç Değerleri Karşılaştırması")
plt.show() 
#boxplot
sns.boxplot(x="Temperature",y="Pressure",data=veri)
plt.xticks(rotation=90)
plt.show()
#countplot
#sıcaklık değerlerinin dağılımı
sns.countplot(x=veri.Temperature)
plt.xticks(rotation=90)
plt.show() 
#pairplot(sıcaklık-basınç ve basınç-sıcaklık grafikleri)
sns.pairplot(data=data2)
plt.show() 
#jointplot
sns.jointplot(data2.Sıcaklık,data2.Basinc,kind="kde",height=8,ratio=6)
plt.xlabel("Sıcaklık")
plt.ylabel("Basınç")
plt.show()
#kde plot(Sıcaklık-Basınç)
sns.kdeplot(data2.Sıcaklık,data2.Basinc,cut=5,shade=True)
plt.show()
#violin plot
sorted_data["Temperature"]=sorted_data["Temperature"]/max(sorted_data["Temperature"])
sorted_data2["Basinc"]=sorted_data2["Basinc"]/max(sorted_data2["Basinc"])
data3=pd.concat([sorted_data["Temperature"],sorted_data2["Basinc"]],axis=1)
sns.violinplot(data=data3,inner="points")
plt.show()
#pie chart
labels=veri.Temperature.value_counts().index[:6]
sizes=veri.Temperature.value_counts().values[:6]
explode=[0,0,0,0,0,0]
colors=["red","yellow","blue","orange","brown","purple"]
plt.figure(figsize=(7,7))
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct="%1.1f%%")
plt.title('En çok tekrarlanan 6 sıcaklık değeri',color = 'blue',fontsize = 15)
plt.show()