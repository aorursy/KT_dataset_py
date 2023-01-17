# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
winedata = pd.read_csv('../input/winemag-data-130k-v2.csv')

winedata.info()
winedata.country.value_counts()
winedata.country.unique()
country_list=list(winedata.country.unique())
country_point_ratio=[]
for i in country_list:
    x=winedata[winedata.country==i]
    if(len(x)>0):
        country_point_rate = sum(x.points)/len(x)
    else:
        country_point_rate=0
    country_point_ratio.append(country_point_rate)
data=pd.DataFrame({"country_list":country_list,"country_point_ratio":country_point_ratio})
new_index=(data["country_point_ratio"].sort_values(ascending=False)).index.values
sorted_data=data.reindex(new_index)
    # visualization
plt.figure(figsize=(20,10))
sns.barplot(x=sorted_data['country_list'], y=sorted_data['country_point_ratio'])
plt.xticks(rotation= 90)#yazıların alt kısma konma açısı 90 desek dik konurdu
plt.xlabel('Countries ')
plt.ylabel('Point Rate')
plt.title('Wine Point Rate Given Countries ')
    
    
winedata.describe()
winedata.price.dropna(inplace=True)
winedata.price.value_counts(dropna=False)
country_list=list(winedata.country.unique())
country_price_ratio=[]
for i in country_list:
    x=winedata[winedata.country==i]
    if(len(x)>0):
        country_price_rate =np.mean(x.price)
    else:
        country_price_rate=0
    country_price_ratio.append(country_price_rate)
data=pd.DataFrame({"country_list":country_list,"country_price_ratio":country_price_ratio})
new_index=(data["country_price_ratio"].sort_values(ascending=False)).index.values
sorted_data2=data.reindex(new_index)
    # visualization
plt.figure(figsize=(12,10))
sns.barplot(x=sorted_data2['country_list'], y=sorted_data2['country_price_ratio'])
plt.xticks(rotation= 90)#yazıların alt kısma konma açısı 90 desek dik konurdu
plt.xlabel('Countries ')
plt.ylabel('Price Rate')
plt.title('Wine Price Rate Given Countries ')
winedata.head()
winedata.taster_name.value_counts(dropna=False)
winedata.taster_name.dropna(inplace=True)
separate = winedata.taster_name.str.split()#stringe çevir ve split yap default'u boşluğa göre böl demek
a,b = zip(*separate)                    
name_list = a+b                         
name_count = Counter(name_list) #sayısını bulur       
most_common_names = name_count.most_common(15) 
x,y = zip(*most_common_names)#!?
x,y = list(x),list(y)

plt.figure(figsize=(15,10))
ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))#renkler için kendi metodu o uzunluk sayısı kadar birbiri ile uyumlu
#farklı renk döndürür
plt.xlabel('Name or Surname tester people')
plt.ylabel('Frequency')
plt.title('Most common 15 Name or Surname of tester people')
sorted_data.head()
sorted_data2.head()
#normalization
sorted_data["country_point_ratio"]=sorted_data["country_point_ratio"]/max(sorted_data["country_point_ratio"])
sorted_data2["country_price_ratio"]=sorted_data2["country_price_ratio"]/max(sorted_data2["country_price_ratio"])
data_concat=pd.concat([sorted_data,sorted_data2["country_price_ratio"]],axis=1)
data_concat.sort_values("country_point_ratio",inplace=True)
 #we del dropna values

#concat
data_concat=pd.concat([sorted_data,sorted_data2["country_price_ratio"]],axis=1)
data_concat.dropna(inplace=True)
data_concat.sort_values("country_point_ratio",inplace=True)

# visualize
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='country_list',y='country_point_ratio',data=data_concat,color='lime',alpha=0.8)#data=data ile seaborn sütunlarımızı tanır ve kolayca x ye ekseni ataması yapabiliriz
sns.pointplot(x='country_list',y='country_price_ratio',data=data_concat,color='red',alpha=0.8)
plt.text(20,0.4,'country price ratio',color='red',fontsize = 17,style = 'italic')
plt.text(20,0.8,'country point ratio',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('Countries',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('country point ratio  VS  country price ratio',fontsize = 20,color='blue')
plt.xticks(rotation= 90)
plt.grid()#kareli göster grafigi #yazılar birbirine girdi?

data_concat.head()

# joint kernel density
# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.
# If it is zero, there is no correlation between variables
# Show the joint distribution using kernel density estimation 
g = sns.jointplot(data_concat.country_point_ratio, data_concat.country_price_ratio, kind="kde", size=7)
plt.savefig('graph.png')
plt.show()
# lmplot 
# Show the results of a linear regression within each dataset
#yani en optimum yerden liner dogru gecer
sns.lmplot(x="country_point_ratio", y="country_price_ratio", data=data_concat)
plt.show()
# Show each distribution with both violins and points
# Use cubehelix to get a custom sequential palette
#farklı future ların içindeki datanın yani değerlerin dağılımına bakmak için kullanılır
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)#googledan bulunabilir renklendirme için
sns.violinplot(data=data_concat, palette=pal, inner="points")#datanın içindeki sayısal şeyleri alıp görselleştirir,inner ile noktalı göterimi yaparız
#histogram tarzı en şişman olduğu yer en çok oradaki datanın df mimizde bulunduğunu ifade eder
plt.show()
data_concat.corr()
#correlation map
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data_concat.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
#annot olayı karelerin içinde rakamlar olsun mu,aradaki line kalınlığı 0.5 ve rengi kırmızı,fmt is eondalikdan sonra yazacağım sayı,ax ax ise
#plotumuz icine headmap i koyucaz demektir
plt.show()
data_concat.head()
# pair plot
sns.pairplot(data_concat)
#only numeric 
plt.show()
winedata.head()
winedata.province.value_counts()
winedata.taster_name.value_counts()
tester = winedata.taster_name.value_counts()
plt.figure(figsize=(10,10))
sns.barplot(x=tester[:7].index,y=tester[:7].values)
plt.title('Most wine tersters',color = 'blue',fontsize=15)
variety = winedata.variety.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=variety[:10].index,y=variety[:10].values)
plt.title('Top 10 wine variety',color = 'blue',fontsize=15)
plt.xticks(rotation= 90)#yazıların alt kısma konma açısı 90 desek dik konurdu