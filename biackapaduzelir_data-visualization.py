# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read datas
median_house_hold_in_come = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv', encoding="windows-1252")
percentage_people_below_poverty_level = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")
percent_over_25_completed_highSchool = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")
share_race_city = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv', encoding="windows-1252")
kill = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0, inplace = True)
#Data setinde anlamsız olan '-' verisini 0.0 la değiştir
percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)
#poverty_rate verimiz bir oran ama veri seti bunu object olarak vermiş bunu sayıya çeviriyoruz
percentage_people_below_poverty_level['Geographic Area'].unique()
area_list = list(percentage_people_below_poverty_level['Geographic Area'].unique())
#percent_people.. datamızın geographic area(eyaletlerini) unique al.list ile listeye cevir area_list listemize depola
area_poverty_ratio = [] #bos bir dizin acıyoruz
for i in area_list: #area_listimizin elemanlarını tek tek geziyoruz
    x= percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area'] == i] #datamın içinde eyaletleri tek tek bul
    area_poverty_rate = sum(x.poverty_rate)/len(x) #poverty rateimin ortalamasını albunu area_poverty_rate diye bir değişkende tut
    area_poverty_ratio.append(area_poverty_rate) #bu area_poverty_rate'i oluşturduğumuz boş dizine ekle
#Bu işlemi neden yaptık ? verilerimizin fakirlik oranını veri setinde verilen şekilde karmaşık değil sıralı sekilde görmek için
data = pd.DataFrame({'area_list': area_list,'area_poverty_ratio': area_poverty_ratio}) #data oluşturuyoruz eyaletleri ve fakirlik oranlarını datamıza ekliyoruz
new_index = (data['area_poverty_ratio'].sort_values(ascending = False)).index.values #bu datamızı sort ediyoruz yani sıralıyoruz. ascending false olması azalan sırada sıralamak, true olsa artan sıra olacaktı
sorted_data = data.reindex(new_index) #bir üst satırda indexleri çekmiştik burada onları datamın yeni indexi yapıyorum

#visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])
plt.xticks(rotation=45) #eyalet isimlerinin kısaltmalarını 45d açıyla yazıyor istersek dik felanda yazdırabiliriz
plt.xlabel('States')
plt.ylabel('Poverty Rate')
plt.title('Poverty Rate Given States')
kill.info()
kill.head()
#kill.name.value_counts()
separate = kill.name[kill.name != 'TK TK'].str.split() #Verisetimde TK TK isimli bir veri var ben bu veriyi istemiyorum. TK TK ya eşit olmayanları al. split ile bosluğa göre ayırır
a,b = zip(*separate) #isim soyisim diye ayrılıyor
name_list = a+b #isim ve soyisimleri bir tuple'a birleştiriyor
name_count = Counter(name_list) #namelerimin sayısını verir
most_common_names = name_count.most_common(15) #most_common metodu ile 15 tane en çok olanı hesaplıyoruz
x,y = zip(*most_common_names) 
x,y = list(x),list(y)

#
plt.figure(figsize=(15,10))
sns.barplot(x=x, y=y, palette =sns.cubehelix_palette(len(x))) # Palette uzunluk sayısı kadar birbiriyle uyumlu farklı renk gösterir
plt.xlabel('Name or Surname of killed People')
plt.ylabel('Frequency')
plt.title('Most common 15 Name or Surname killed of people')
percent_over_25_completed_highSchool.info()
percent_over_25_completed_highSchool.head()
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()
percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0, inplace= True)
percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype('float')
percent_over_25_completed_highSchool['Geographic Area'].unique()
area_list = list(percent_over_25_completed_highSchool['Geographic Area'].unique())
area_highschool=[]
for i in area_list:
    x=percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['Geographic Area']==i]
    area_highschool_rate = sum(x.percent_completed_hs)/len(x)
    area_highschool.append(area_highschool_rate)

#sorthing
data = pd.DataFrame({'area_list': area_list, 'area_highschool_ratio': area_highschool})
new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values
sorted_data2 =data.reindex(new_index)

#Visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['area_highschool_ratio'])
plt.xticks(rotation=45)
plt.xlabel('States')
plt.ylabel('High School Graduate Rate')
plt.title('Percentage of Given States Population Above 25 that has Graduated High School')
share_race_city.info()
share_race_city.replace(['-'],0.0,inplace=True)
share_race_city.replace(['(X)'],0.0,inplace=True)
share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)
area_list = list(share_race_city['Geographic area'].unique())
share_white = []
share_black = []
share_native_american = []
share_asian = []
share_hispanic = []
for i in area_list:
    x = share_race_city[share_race_city['Geographic area']==i]
    share_white.append(sum(x.share_white)/len(x))
    share_black.append(sum(x.share_black)/len(x))
    share_native_american.append(sum(x.share_native_american)/len(x))
    share_asian.append(sum(x.share_asian)/len(x))
    share_hispanic.append(sum(x.share_hispanic)/len(x))
    
#Visualization

f,ax = plt.subplots(figsize=(10,15))
sns.barplot(x=share_white, y=area_list, color='lime', alpha=0.5, label='White')
sns.barplot(x=share_black, y=area_list, color='blue', alpha=0.7, label='Black')
sns.barplot(x=share_native_american, y=area_list, color='cyan', alpha=0.6, label='Native American')
sns.barplot(x=share_asian, y=area_list, color='yellow', alpha=0.6, label='Asian')
sns.barplot(x=share_hispanic, y=area_list, color='red', alpha=0.6, label='Hispanic')

ax.legend(loc='best', frameon=True) #legendin görünürlügü
ax.set(xlabel = 'Percentage of Rages', ylabel='States', title='Percentage of States Population According to Races')
sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max(sorted_data['area_poverty_ratio']) #Normalizasyon yapıyoruz
sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max( sorted_data2['area_highschool_ratio'])
data = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1) #iki datayı columlarla birleştiriyorum. sorted_dataya area_poverty_ratio yazmadım çünkü ama sorted_data içinde area_list ve poverty_ratio var, bu iki sütuna highschool_ratio de eklendi, yani ek olarak eyaleti ekledik.
data.sort_values('area_poverty_ratio',inplace =True)

#visualize

f,ax1 = plt.subplots(figsize=(20,10))
sns.pointplot(x='area_list', y='area_poverty_ratio', data=data, color='lime', alpha=0.8)
sns.pointplot(x='area_list', y='area_highschool_ratio', data=data, color='red', alpha=0.8)
plt.text(40,0.6,'high school graduate ratio', color='red', fontsize = 17, style= 'italic')
plt.text(40,0.55,'poverty ratio',color='lime', fontsize= 18, style='italic')
plt.xlabel('States', fontsize=25, color='blue')
plt.ylabel('Values', fontsize=25, color='blue')
plt.title('High School Graduate VS Poverty Rate',fontsize=20, color='blue')
plt.grid()

data.head()
#pearsonr = iki feature arasında correlation'u gösterir. 1 se pozitif -1 se negatif correlation'u gösterir. Eğer 0 ise burada correlation yoktur.
#Eger 1 e yakında pozitife yaklasan, -1 se negatife yaklasan. ama %100 değil
g =sns.jointplot(data.area_poverty_ratio,data.area_highschool_ratio, kind='kde', size=7)
plt.savefig('graph.png') #resmin kaggleda görülebilmesi için/dersle ilgisi yok
plt.show()
# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }
g = sns.jointplot('area_poverty_ratio','area_highschool_ratio',data=data, size=5, ratio=3, color='red')
#data=data yaptığımız zaman bu bizim hangi datayla çalışacağımızı anlıyor. 'area_poverty_ratio','area_highschool_ratio' biliyorum ve string halinde yazabilirim
kill.race.head(5)
kill.race.dropna(inplace= True) #datamda boş ırk veya saçma sapan data varsa datadan çıkar 
labels = kill.race.value_counts().index #hangi ırktan kaç tane var
colors =['grey','blue','red','yellow','green','brown'] #pychardımın renkleri
explode =[0,0,0,0,0,0]# pychartlarımın oranı. bu listeyi daha sonra dolduracaz
sizes =kill.race.value_counts().values

#visual
plt.figure(figsize=(7,7))
plt.pie(sizes, explode=explode,labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Killed People According to Races', color='blue', fontsize=15)
#lmplot - diger plotların farkı lineer regresyon
sns.lmplot(x='area_poverty_ratio', y='area_highschool_ratio', data=data)
plt.show()
sns.kdeplot(data.area_poverty_ratio, data.area_highschool_ratio,color='darkgreen', shade=True, cut=5)
plt.show()
#ikisi arasındaki ilişkiye değil farklı featureların içindeki datanın(değerlerinin) dağılımına bakıyor. sayısal şeyleri alıp görselleştirir 
pal =sns.cubehelix_palette(2,rot=-.5,dark=.3)
sns.violinplot(data=data, palette=pal, inner='points') 
plt.show()
#correlation map- heatmap
data.corr()
f,ax= plt.subplots(figsize=(5,5))
sns.heatmap(data.corr(),annot=True, linewidths=.5, linecolor='red', fmt='.1f',ax=ax)
plt.show()
kill.head()
sns.boxplot(x='gender',y='age', hue= 'manner_of_death', data=kill, palette='PRGn')