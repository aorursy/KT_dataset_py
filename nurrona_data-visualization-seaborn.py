# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from collections import Counter

%matplotlib inline 



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#verileri okuttuk

median = pd.read_csv('../input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv' , encoding = "windows-1252")

percentage = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv'  , encoding = "windows-1252")

percentover = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding = "windows-1252")

share = pd.read_csv('../input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv', encoding = "windows-1252")

kill = pd.read_csv('../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv' , encoding = "windows-1252")
# ilk 5 eyaletin fakirlik oranı

percentage.head()
#data hakkında bigi verir

percentage.info()
# değerlerin toplam sayısını verir

percentage.poverty_rate.value_counts()
# başına len komutu eklendiğinde çıkan sonucun toplamını verir

# eyalet sayısını verir

len(percentage['Geographic Area'].unique())
# .unıque ile belirtilen alanda kaç farklı değer olduğunu verir

# bu veri setinde eyaletleri verdi

percentage['Geographic Area'].unique()
# her eyaletin fakirlik oranını büyükten küçüğe doğru sıraladık

percentage.poverty_rate.replace(['-'], 0.0, inplace = True) # değer olarak '-' olanların yerine 0.0 atadık

percentage.poverty_rate = percentage.poverty_rate.astype(float)

area_list = list(percentage['Geographic Area'].unique())

area_poverty_ratio = []

for i in area_list:

    x = percentage[percentage['Geographic Area'] == i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

data = pd.DataFrame({'area_list':area_list,'area_poverty_ratio':area_poverty_ratio})

new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)





# visualization (görselleştirme)

plt.figure(figsize=(15,10))

sns.barplot(x = sorted_data['area_list'], y = sorted_data['area_poverty_ratio'])

plt.xticks(rotation = 45) # x axisisine(eksenine) 45 derecelik açı ile yerleştirdik

plt.xlabel('States')

plt.ylabel ('Poverty Rate')

plt.title('Poverty Rate Given States')

plt.show()
kill.name.value_counts()
# öldürülen insanların isim ve soyisimlerinden en çok kullanılan 15 tanesini bulduk

separate = kill.name[kill.name != 'TK TK'].str.split() # tk tk ismini dahil etmedik

a,b = zip(*separate) # isim ve soyisimleri ayırdık

name_list = a + b # isim ve soyisimleri birleştirdik

name_count = Counter(name_list)

most_common_names = name_count.most_common(15) 

x , y = zip(*most_common_names)

x , y = list(x), list(y)



#



plt.figure(figsize=(15,10))

sns.barplot(x = x , y = y , palette=sns.cubehelix_palette(len(x))) # palette fonksiyonu birbiri ile uyumlu 15 rengi verir

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Fruquency')

plt.title('Most commob 15 Name or Surname of killed people')

plt.show()
percentover.percent_completed_hs.value_counts()
# eyeletlerdeki, 25 yaşından büyük insanların liseden mezun olma oranını bulduk 

percentover.percent_completed_hs.replace(['-'], 0.0, inplace=True)

percentover.percent_completed_hs = percentover.percent_completed_hs.astype(float)

area_list = list(percentover['Geographic Area'].unique())

area_highschool = []

for i in area_list:

    x = percentover[percentover['Geographic Area'] == i ]

    area_highschool_rate = sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)

    

    

# sorting (sıralama)

data = pd.DataFrame({'area_list': area_list, 'area_highschool_ratio' : area_highschool})

new_index = (data['area_highschool_ratio'].sort_values(ascending = True)).index.values #indexini alıp azalandan artana göre sıraladık

sorted_data2 = data.reindex(new_index)





# visualization (görselleştirme)

plt.figure(figsize=(15,10))

sns.barplot(x = sorted_data2['area_list'], y = sorted_data2['area_highschool_ratio'])

plt.xticks(rotation = 90) # x eksenindeki yazıların hangi açıyla duracağını ayarladık

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")

plt.show()
share.info()
# Irklara göre nüfusu sıraladık

share.replace(['-'], 0.0, inplace = True)

share.replace(['(X)'], 0.0, inplace = True)

share.loc[:, ['share_white', 'share_black', 'share_native_american', 'share_asian', 'share_hispanic']] = share.loc[:,['share_white', 'share_black', 'share_native_american', 'share_asian', 'share_hispanic']].astype(float)

area_list = list(share['Geographic area'].unique())

share_white = []

share_black = []

share_native_american = []

share_asian = []

share_hispanic = []

for i in area_list:

    x = share[share['Geographic area'] == i]

    share_white.append(sum(x.share_white) / len(x))

    share_black.append(sum(x.share_black) / len(x))

    share_native_american.append(sum(x.share_native_american) / len(x))

    share_asian.append(sum(x.share_asian) / len(x)) 

    share_hispanic.append(sum(x.share_hispanic) / len(x))

    

    

f, ax = plt.subplots(figsize = (9,15))

sns.barplot(x = share_white, y =area_list, color = 'pink', alpha = 0.5, label = 'White') 

sns.barplot(x = share_black, y =area_list, color = 'blue', alpha = 0.5, label = 'African American')

sns.barplot(x = share_native_american, y =area_list, color = 'purple', alpha = 0.5, label = 'Native American') 

sns.barplot(x = share_asian, y =area_list, color = 'yellow', alpha = 0.5, label = 'Asian')

sns.barplot(x = share_hispanic, y =area_list, color = 'red', alpha = 0.5, label = 'Hispanic') 





ax.legend(loc = 'lower right', frameon = True) # frameon = True: altta bilgi veren kutucuğun çevresinde çerçeve oluşmasını sağlar

ax.set(xlabel = 'Percentage of Races', ylabel = 'States', title = "Percentage of State's Population Acording to Races")

plt.show()
# eyaletler arasındaki liseden mezun olma ve fakirlik oranını karşılaştırdık

sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio'] / max (sorted_data['area_poverty_ratio'])

sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio'] / max (sorted_data2['area_highschool_ratio'])

data = pd.concat([sorted_data, sorted_data2['area_highschool_ratio']], axis = 1)

data.sort_values('area_poverty_ratio', inplace = True)





# visualize

f, ax = plt.subplots(figsize = (20,10))

sns.pointplot(x ='area_list', y= 'area_poverty_ratio', data = data, color ='pink', alpha = 0.8)

sns.pointplot(x ='area_list', y= 'area_highschool_ratio', data = data, color ='purple', alpha = 0.8)

plt.text(40, 0.6, 'high school graduate ratio', color = 'purple', fontsize = 17, style = 'italic')

plt.text(40, 0.55, 'poverty ratio', color = 'pink', fontsize = 18, style='italic')

plt.xlabel('States', fontsize = 15, color = 'blue')

plt.ylabel('Values', fontsize = 15, color = 'blue')

plt.title('High School Graduate vs Poverty Rate', fontsize = 25, color = 'blue')

plt.grid() #arka plandaki çizgilerin belirmesini sağlar(kafes)

plt.show()
data.head()
# eyaletler arasındaki liseden mezun olma ve fakirlik oranını karşılaştırdık

# kde : kernel density estimation (çekirdek yoğunluğu tahmini)

g = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind="kde", size = 7)
# eyaletlerin yoksulluk oranına göre liseden mezun olma oranını bulduk

g = sns.jointplot('area_poverty_ratio','area_highschool_ratio', data = data, size = 5, ratio = 3, color ='r')
kill.race.value_counts()
# Öldürülen insanların ırklarının oranını bulduk

kill.race.dropna(inplace=True)

labels = kill.race.value_counts().index

colors = ['beige', 'blue', 'pink', 'yellow', 'purple', 'orange']

explode = [0,0,0,0,0,0]

sizes = kill.race.value_counts().values



#visual

plt.figure(figsize =(7,7))

plt.pie(sizes, explode = explode, labels = labels, colors = colors, autopct = '%1.1f%%')

plt.title('Killed People According to Races', color = 'blue', fontsize = 15)

plt.show()
# Fakirlik oranı ile liseden mezun olma oranını karşılaştırdık

# Fakirlik oranı arttıkça liseden mezun olma oranının azaldığını gördük

sns.lmplot(x ='area_poverty_ratio', y = 'area_highschool_ratio', data = data)

plt.show()
# liseden mezun olma oranına göre yoksulluk

# shade = true: çizimin içini doldurur

sns.kdeplot(data.area_poverty_ratio, data.area_highschool_ratio, shade=True, cut = 5)

plt.show()
# inner = points : data içindeki noktaları göstermeye yarar, içindeki noktalar data içindeki değerler

pal = sns.cubehelix_palette(2, rot = -.2, dark=.3)

sns.violinplot(data = data, palette = pal, inner= 'points')

plt.show()
#correlation map

f, ax = plt.subplots(figsize= (5,5))

sns.heatmap(data.corr(), annot=True, linewidths=.5, linecolor='orange', fmt='.1f', ax =ax)

plt.show()
# insanın nasıl öldürüldüğünü gösterir

# aynıı zamanda öldürülen insanların yaşı ve cinsiyetini öğreniriz

# tabloda üstte görünen nokta benzeri şeyler istisnaları gösterir

sns.boxplot(x = 'gender', y = 'age', hue= 'manner_of_death', data = kill, palette='PRGn')

plt.show()
sns.swarmplot(x = 'gender', y = 'age', hue = 'manner_of_death', data = kill)

plt.show()
# fakirlik oranı ile liseden mezun olma oranını karşılaştırdık

sns.pairplot(data)

plt.show()
kill.gender.value_counts()
sns.countplot(kill.gender)

plt.title('gender', color = 'black', fontsize = 15)

plt.show()
armed = kill.armed.value_counts()

print(armed)

plt.figure(figsize = (10,7))

sns.barplot(x = armed[:7].index, y = armed[:7].values)

plt.ylabel('Number of Weapon')

plt.xlabel('Weapon Types')

plt.title('Kill weapon', color ='blue', fontsize = 15 )

plt.show()
# 25 yaşından büyük veya küçük olup öldürülen insanları karşılaştırdık

above25 = ['above25' if i >= 25 else 'below25' for i  in kill.age]

df = pd.DataFrame({'age':above25})

sns.countplot(x = df.age)

plt.ylabel('Number of Killed People')

plt.title('Age of killed people', color = 'blue', fontsize = 15)

plt.show()
# öldürülen insanların ırkına göre sayısını görüntüledik

sns.countplot(data = kill, x = 'race')

plt.title('Race of killed people', color = 'blue', fontsize =15)

plt.show()
# şehirlerin tehlike oranları

city = kill.city.value_counts()

plt.figure(figsize=(10, 7))

sns.barplot(x = city[:12].index, y = city[:12].values)

plt.xticks(rotation = 45)

plt.title('Most dangerous cities', color = 'blue', fontsize = 15)

plt.show()