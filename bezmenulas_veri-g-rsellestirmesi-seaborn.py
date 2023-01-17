# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read datas

median_house_hold_in_come = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding="windows-1252")



percentage_people_below_poverty_level = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")



percent_over_25_completed_highSchool = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")



share_race_city = pd.read_csv('../input/ShareRaceByCity.csv', encoding="windows-1252")



kill = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
# poverty_rate içinde saçma bir değer var mı??

percentage_people_below_poverty_level.poverty_rate.value_counts()



# 201 tane belirtilmemiş(-) değer var.
percentage_people_below_poverty_level['Geographic Area'].unique()
# Poverty rate of each state (Her devletin yoksulluk oranı)



percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace = True)

# Datada bulunan (-) işaretin karşısında 201 adet var ama ne olduğunu bilmiyoruz.

# Bu nedenle 0 yapıcaz.



percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)

# poverty_rate ---> 29329 non-null object kısmı float yapmalıyız çünkü sayısal bir değer



area_list = list(percentage_people_below_poverty_level['Geographic Area'].unique())

#Benden şehirlerdeki yoksulluğu istediği için her şehri area_list içine gönderdim



area_poverty_ratio = []

for i in area_list:

    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]

    # her Coğrafi alanın bilgilerini sırasıyla x'in içine attıyoruz.

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    # Daha sonra bölgedeki yoksulluk oranının toplamı ile kaç değer varsa bölüp oranı buluyoruz.

    area_poverty_ratio.append(area_poverty_rate)



# Ben görselleştirmeyi rastgele değil düzenli yapmak istediğim için yoksulluğu büyükten küçüğe sıralıyorum.

# Böylece hem görsel açıdan hem de veriyi anlama açısından kazanç sağlıyorum.



data = pd.DataFrame({'area_list': area_list,'area_poverty_ratio':area_poverty_ratio})

# bölgeleri ve bu bölgeleride ki oranları dataFrame olarak kullanıyoruz.



new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values

# sort_values(ascending=False) kısmı azalan sırada sıralıyor.

# inde.values  ==>  indexleri al diyor ama indexi alırsan array olarak alacak bunun da value yani değerlerini al diyor. 



sorted_data = data.reindex(new_index) # ????????????????????



# visualization

plt.figure(figsize=(15,15))

sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])



# burada => ax=sns.barplot... ax eşitlemiştik ama şimdilik gerek yok.

plt.xticks(rotation= 45) # x eksenindeki yazılar 45 derecelik açıyla yazdı

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')
kill.head()
kill.name.value_counts()



# TK TK ---> 49 /// TK TK adında 49 kişi bunları kaldırıcam
# En yaygın 15 Ölen insanın adı veya soyadı



separate = kill.name[kill.name != "TK TK"].str.split()

# split ile ad ve soyadı ayırdık



a,b = zip(*separate) # unzip

name_list = a+b

# bu zip ve toplama olayına bak örnek = ["ali", "haydar"]



name_count  = Counter(name_list)

# her isimden kaç tane var onu bulduk



most_common_names = name_count.most_common(15)

# en çok olan 15 taneyi bul



x,y = zip(*most_common_names) # unzip

x,y = list(x),list(y)



plt.figure(figsize=(15,10))

ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))

# palette ---> renk veriyor fakat aynı rengin uygun tonlarında



plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most common 15 Name or Surname of killed people')
# En yaygın 15 isim ve soyismi grafiklerini subplot yardımıyla çizdirmek.



separate = kill.name[kill.name != "TK TK"].str.split()

# split ile ad ve soyadı ayırdık



name,surname = zip(*separate)



nameCount = Counter(name)

surnameCount = Counter(surname)



mostCommonName = nameCount.most_common(15)

mostCommonSurname = surnameCount.most_common(15)



xName,yName = zip(*mostCommonName) # unzip

xName,yName = list(xName),list(yName)



xSurname, ySurname = zip(*mostCommonSurname)

xSurname, ySurname = list(xSurname), list(ySurname)



plt.figure(figsize=(15,10))

plt.subplot(2,1,1)

ax= sns.barplot(x=xName, y=yName,palette = sns.cubehelix_palette(len(xName)))

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most common 15 Name of killed people')





plt.figure(figsize=(15,10))

plt.subplot(2,1,2)

ax= sns.barplot(x=xSurname, y=ySurname,palette = sns.cubehelix_palette(len(xSurname)))

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most common 15 Surname of killed people')
percent_over_25_completed_highSchool.head()
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()

# 197 tane tanımsız (-) var. Onları sıfır yapıcaz.
percent_over_25_completed_highSchool.info()

# percent_completed_hs ---> float yapmalıyız 
# Eyaletlerde 25 yaşından büyük nüfusun lise mezuniyet oranı

# High school graduation rate of the population that is older than 25 in states



percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0,inplace = True)

percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)

area_list = list(percent_over_25_completed_highSchool['Geographic Area'].unique())

area_highschool = []

for i in area_list:

    x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['Geographic Area']==i]

    area_highschool_rate = sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)

# sorting

data = pd.DataFrame({'area_list': area_list,'area_highschool_ratio':area_highschool})

new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values

sorted_data2 = data.reindex(new_index)

# Burada datayı sıraladık.



# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['area_highschool_ratio'])

plt.xticks(rotation= 90)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")
share_race_city.head()
share_race_city.info()
x = share_race_city[share_race_city.share_white == "(X)"]

x
# Siyah, beyaz, kızılderili, asyalı ve İspanyol olan ırklara göre devlet nüfusunun yüzdesi

# Percentage of state's population according to races that are black,white,native american, asian and hispanic



share_race_city.replace(['-'],0.0,inplace=True)

share_race_city.replace(['(X)'],0.0,inplace=True)

# Veriyi temizledik.



share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)

# Veri sayısal olmasına rağmen str olarak bize verildi bu veriyi kullanmak için float'a çevirdik.



area_list = list(share_race_city['Geographic area'].unique())

share_white = []

share_black = []

share_native_american = []

share_asian = []

share_hispanic = []

# Irkları ayırdık.



for i in area_list:

    x = share_race_city[share_race_city['Geographic area']==i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black) / len(x))

    share_native_american.append(sum(x.share_native_american) / len(x))

    share_asian.append(sum(x.share_asian) / len(x))

    share_hispanic.append(sum(x.share_hispanic) / len(x))

# Irkların değerlerini bulduk.



# visualization

f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=share_white,y=area_list,color='green',alpha = 0.5,label='White' )

sns.barplot(x=share_black,y=area_list,color='blue',alpha = 0.7,label='African American')

sns.barplot(x=share_native_american,y=area_list,color='cyan',alpha = 0.6,label='Native American')

sns.barplot(x=share_asian,y=area_list,color='yellow',alpha = 0.6,label='Asian')

sns.barplot(x=share_hispanic,y=area_list,color='red',alpha = 0.6,label='Hispanic')



ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu

ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races")
sorted_data.head()
sorted_data2.head()
sorted_data["area_poverty_ratio"] = sorted_data["area_poverty_ratio"]/max(sorted_data["area_poverty_ratio"])

sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max(sorted_data2['area_highschool_ratio'])

# normalize ettik.



data = pd.concat([sorted_data,sorted_data2["area_highschool_ratio"]], axis=1)

# sorted_data2["area_highschool_ratio"] sutununu sorted_data ile birleştirdik.

# biz aslında sorunun cevabını bulduk fakat ayrıca görselleştirmek istiyoruz.



data.sort_values("area_poverty_ratio",inplace=True)

# sıraladık.
data.head() # görselleştiriceğimiz data
f,ax1 = plt.subplots(figsize=(20,10))

sns.pointplot(x='area_list', y='area_poverty_ratio', data=data, color='lime', alpha=0.8)

# x='area_list' diye yazabildik çünkü seaborn data içi parametresi ile görebiliyor.

# x = data["area_list"] yazmamıza gerek yok.

sns.pointplot(x="area_list", y="area_highschool_ratio", data=data, color="red", alpha=0.8)

plt.text(40,0.6, "high school graduate ratio",color="red", fontsize=17, style="italic")

plt.text(40,0.55, "poverty ratio", color="lime", fontsize=18, style="italic")

plt.xlabel("States",fontsize=15, color="blue")

plt.ylabel("Values",fontsize=15, color="blue")

plt.title("High School Graduation Rate vs Poverty Rate", fontsize=20,color="blue")

plt.grid()
# Lise mezuniyet oranına karşı vs Her devletin yoksulluk oranı ile farklı seaborn kodu stili olan görselleştirme



g = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind="kde", size=7)

plt.savefig("graph.png") # kaggle'da görüntü için yazdık.

plt.show()



# x ekseni area_poverty_ratio , y ekseni area_highschool_ratio

# kind="kde" => kde açılımı joint kernel density (ortak çekirdek yoğunluğu)

# pearsonr => 1 ise pozitif korelasyon var ve -1 ise negatif korelasyon var.

# Sıfır ise değişkenler arasında korelasyon yoktur.

# burda tam olmasada negatif korelsyon (ters orantı) var
# you can change parameters of joint plot

# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }

# Different usage of parameters but same plot with previous one



g = sns.jointplot("area_poverty_ratio", "area_highschool_ratio", data=data,size=5, ratio=3, color="red", kind="scatter")

# ratio alanı ayarlıyor.

plt.show()
g = sns.jointplot("area_poverty_ratio","area_highschool_ratio", data=data,size=5, color="blue", ratio=3, kind="hex")
kill.race.head(10)
kill.race.value_counts()

# o = other yani listede olmyan ırklar
# Race rates according in kill data (ırk oranları)



kill.race.dropna(inplace = True)

# boş varsa data çıkardık

labels = kill.race.value_counts().index # sadece indexleri aldık

colors = ["grey","blue","red","yellow","green","brown"]

explode = [0,0,0,0,0,0] # oranlar

sizes = kill.race.value_counts().values # değerler (values)



# visual

plt.figure(figsize=(7,7))

# NOT = pie plot bir seaborn kütüphnesinin değil matplot'un

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%1.1f%%")

plt.title('Killed People According to Races',color = 'blue',fontsize = 15)
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# lmplot 

# Show the results of a linear regression within each dataset



# Lise mezuniyet oranının görselleştirilmesi - Her devletin yoksulluk oranı

# lmplot

# Her veri kümesindeki doğrusal regresyon sonuçlarını göster



sns.lmplot(x="area_poverty_ratio", y="area_highschool_ratio", data=data)

plt.show()



# line(çizgi) en optimum noktadan geçmiş. bu bir makine öğrenimine örnek
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# cubehelix plot



sns.kdeplot(data.area_poverty_ratio, data.area_highschool_ratio, shade=True, cut=5)

plt.show()



# shde = dalga içlerinin dolu olup olmamasını belirler

# cut = büyüklüğü
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=data ,palette=pal, inner="points")

plt.show()



# inner="points" --> ortadaki noktalar
data.corr()
f,ax = plt.subplots(figsize=(6, 6)) 

sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()
kill.head()
kill.manner_of_death.unique()
# manner of death(olum sekli) : ates edilerek, ates edilerek ve sok tabancasiyla

# gender cinsiyet

# age: yas

# Plot the orbital period with horizontal boxes (Yörünge periyodunu yatay kutularla çizin)



sns.boxplot(x="gender", y="age", hue="manner_of_death", data=kill, palette="PRGn")

plt.show()
# swarm plot

# manner of death(olum sekli) : ates edilerek, ates edilerek ve sok tabancasiyla

# gender cinsiyet

# age: yas



sns.swarmplot(x="gender", y="age",hue="manner_of_death", data=kill)

plt.show()



# yüksek veri olduğunda çizdirmek zordur
sns.pairplot(data)

plt.show()
kill.gender.value_counts()
kill.head()
# kill properties

# Manner of death



sns.countplot(kill.gender)

#sns.countplot(kill.manner_of_death)

plt.title("gender", color="blue", fontsize=15)
# kill weapon

armed = kill.armed.value_counts()

#print(armed)

sns.barplot(x = armed[:7].index, y=armed[:7].values)

plt.ylabel('Number of Weapon')

plt.xlabel('Weapon Types')

plt.title('Kill weapon',color = 'blue',fontsize=15)
# age of killed people

above25 = ["above25" if i>=25 else "below25" for i in kill.age]

df = pd.DataFrame({"age":above25})

sns.countplot(x=df.age)
# Race of killed people

sns.countplot(data=kill, x='race')

plt.title('Race of killed people',color = 'blue',fontsize=15)
# Most dangerous cities

city = kill.city.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=city[:12].index,y=city[:12].values)

plt.xticks(rotation=45)

plt.title('Most dangerous cities',color = 'blue',fontsize=15)
# most dangerous states

state = kill.state.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=state[:20].index,y=state[:20].values)

plt.title('Most dangerous state',color = 'blue',fontsize=15)
# Having mental ilness or not for killed people

sns.countplot(kill.signs_of_mental_illness)

plt.xlabel('Mental illness')

plt.ylabel('Number of Mental illness')

plt.title('Having mental illness or not',color = 'blue', fontsize = 15)
# Threat types

sns.countplot(kill.threat_level)

plt.xlabel('Threat Types')

plt.title('Threat types',color = 'blue', fontsize = 15)
# Flee types

sns.countplot(kill.flee)

plt.xlabel('Flee Types')

plt.title('Flee types',color = 'blue', fontsize = 15)
# Having body cameras or not for police

sns.countplot(kill.body_camera)

plt.xlabel('Having Body Cameras')

plt.title('Having body cameras or not on Police',color = 'blue',fontsize = 15)
# Kill numbers from states in kill data

sta = kill.state.value_counts().index[:10]

sns.barplot(x=sta,y = kill.state.value_counts().values[:10])

plt.title('Kill Numbers from States',color = 'blue',fontsize=15)