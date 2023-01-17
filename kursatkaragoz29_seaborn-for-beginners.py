import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline



from subprocess import check_output

print("DATASETS NAME:")

print(check_output(["ls", "../input"]).decode("utf8")) #fatal-police-shootings-in-the-us



import os

print("DATASETS")

for dirname, _, filenames in os.walk('/kaggle/input'):    

    for filename in filenames:

        print(os.path.join(dirname, filename)) #directorys datasets



# Any results you write to the current directory are saved as output.
median_house_hold_in_come = pd.read_csv("../input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv",encoding="windows-1252")

percent_over_25_completed_highSchool = pd.read_csv("../input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv",encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv("../input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv",encoding="windows-1252")

kill = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv",encoding="windows-1252")

share_race_city = pd.read_csv("../input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv",encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()



# poverty_rate = fakirlik oranını vermektedir fakat object type sinde olduğu görülmektedir.

# Bu datanın typesini float olarak güncellemeliyiz.
#Geographic Area uniq yapılarak, Amerikadaki eyaletlere ulaşmış olduk

percentage_people_below_poverty_level['Geographic Area'].unique()
#value_counts ile fakirlik oranına (poverty_rate) göre grupladık

percentage_people_below_poverty_level.poverty_rate.value_counts()
#-----------------------------------Povert rate ıf each state (Her bir eyaletin fakirlik oranı ?) BARPLOT İLE GÖRSELLEŞTİRME-----------------------------------------------

##################### DATA ONARMA ##############

# Fakirlik oranı(poverty_rate) "-" olan sampleları 0.0 ile değiştirdik ve 

# inplace ile yapılan işlemi benim variablema kaydet demiş olduk.

percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace = True)

#poverty_rate kolonunun tipini float olarak güncelledik. #(object => float)

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)

###################--------------------##########



#Amerikadaki her bir eyalet 

area_list = list(percentage_people_below_poverty_level['Geographic Area'].unique())





area_poverty_ratio=[]  #Her bir eyaletin fakirlik oranını tutmak için bir liste  oluşturduk

for i in area_list:

    # area_list e kaydedilmiş uniq değerlerin her bir tek tek bulunuyor ve x' eşitleniyor

    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area'] == i] #uniq'e eşit tüm eyaletlerin poverty_rate bilgileri x içerisinde.

    area_poverty_rate = sum(x.poverty_rate) / len(x)  # ilgili uniq'e ait povert_ratelerin ortalamasını aldık

    area_poverty_ratio.append(area_poverty_rate) # listeye ekledik

    

# Yukarıda tüm eyaletlerin fakirlik oranını (area_poverty_ratio) bulmuş olduk.

# Şimdik yapacağımız işlem bir dateframe oluşturmak, bu dataframe'de eyaletler karşılığı olarakta fakirlik oranları yer almalıdır. 

# Aynı zamanda fakirlik oranlarına bağlı olarak büyükten kücüğe sıralayarak görselleştirme işleminde daha hoş bir sonuç alabiliriz.



# Yeni bir DataFrame oluşturduk => kolonlarını ve valuelerini verdik. 

# Kolonlar:"area_list",area_poverty_ratio, Values: area_list(list),area_poverty_ration(list)

data = pd.DataFrame({"area_list" : area_list, "area_poverty_ratio":area_poverty_ratio}) 

# data isimli DataFrame şuan hazır fakat büyük küçüğe sıralı değil ve sıralandığı taktirde indeksleri değişecek.

# Aşağıda verileri(fakirlik oranlarına göre) büyükten kücüğe sıraladık ve oluşan indekslerini çektik

new_index = (data['area_poverty_ratio'].sort_values(ascending = False)).index.values 

sorted_data = data.reindex(new_index) # yeni indekslemeyi dataya gömdük reindex(indekslemeyi biçimlendirmek için kullanıldı)

#Sıralı veri sorted_data değişkenine aktarıldı

# Visualization



plt.figure(figsize=(15,10))  # yeni bir figüre açtık boyutu 15'e 10 yaptık. (x,y)

sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio']) # barplot çizdirdi x=eyaletler , y= fakirlik oranı

plt.xticks(rotation = 45) # x eksenindeki isimleri 90'derecelik açı ile koy

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Povert Rate Given States') 

plt.show()
kill.head()
kill.name.value_counts()
# kill.name içerisinde tktk olmayan isimli alacağız diğerleri kalacak



separate = kill.name[kill.name != 'TK TK'].str.split()

a,b = zip(*separate) #name' i TK TK olmayanları boşluğa göre (isim ve soyisim) ayırdık ve a=isim,b=soyisim aktardık

name_list = a+b



name_count = Counter(name_list) #Her isimden kaç adet olduğunu hesapladık

most_common_names = name_count.most_common(15) # most_common methodu ile en çok 15'ini elde ettik.



x,y = zip(*most_common_names) # tekrar unzip yaparak isimleri ve öldürülme sayılarını ayrı elde ediyoruz

x,y = list(x) , list(y)  #  x==>name ,  y==>kill_count



## Visulization

plt.figure(figsize=(15,10))

#palette: çubukların renklerini ifade eder verilen uzunluk parametresi kadar birbirine yakın ama farklı renk üretir

ax = sns.barplot(x=x , y=y, palette=sns.cubehelix_palette(len(x))) 

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most common 15 Name or Surname of killed people')

plt.show()
percent_over_25_completed_highSchool.head()

# Eyaletler, Şehirler, liseden mezun olma yaş ortalaması
percent_over_25_completed_highSchool.info()

#percent_completed_hs lisen mezun olma yaş ortalaması sayısal bir tipe çevrilmelidir
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()

# percent_completed_hs incelediğimizde anlamsız "-" değerinden 197 adet olduğunu görüyoruz

# bu sampleların data'dan çıkarılması ya da 0'a çevrilmesi gerekir.
#percent_completed_hs kolonundaki anlamsız ("-") değerleri 0.0'a eşitle ve kaydet.

percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0,inplace=True)

# string tipinde olan percent_completed_hs'i float tipine çevirdik

percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)



area_list = list(percent_over_25_completed_highSchool['Geographic Area'].unique())

area_highschool = []

for i in area_list:

    x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['Geographic Area'] == i] # unique eyaletler filtrelenerek x'e aktarıldı.

    area_highschool_rate = sum(x.percent_completed_hs) / len(x)

    area_highschool.append(area_highschool_rate)

data = pd.DataFrame({'area_list':area_list, 'area_highschool_ratio':area_highschool})

new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values

sorted_data2 = data.reindex(new_index)



#visualization

plt.figure(figsize=(15,10))

ax= sns.barplot(x=sorted_data2['area_list'],y=sorted_data2['area_highschool_ratio'])

plt.xticks(rotation=90)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")

plt.show()
share_race_city.head()
share_race_city.info()

#Sayısal olması gereken değerler object olarak görünmektedir sayısal tipe çevrilmelidir.
# Percentageof state's population according to races that are black,white,native american, asian and hispanic

#anlamsız "-" ve ('x') değerler 0'a eşitlendi

share_race_city.replace(['-'],0.0,inplace = True)

share_race_city.replace(['(X)'],0.0,inplace = True)

# sayısal olması gereken object(string) değerler sayısal (foat) değere çevriliyor

share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float) 

area_list = list(share_race_city['Geographic area'].unique())

share_white=[]

share_black=[]

share_native_american=[]

share_asian=[]

share_hispanic=[]



filter1 = share_race_city['Geographic area']



for i in area_list:

    # x içerisinde i'nci eyalet için beyaz,siyah,yerli,asyalı ve ispanyol kökenlilerin oranlarını dizilere aktarıyoruz.

    x = share_race_city[filter1 == i] # share_race_city içinde geographic lerden uniq değerlerimize eşit olanları x'e aktardık

    share_white.append(sum(x.share_white) / len(x)) # i'nci eyaletteki beyazların oranı

    share_black.append(sum(x.share_black) / len(x)) # i'nci eyaletteki siyahların oranı

    share_native_american.append(sum(x.share_native_american) / len(x)) # i'nci eyaletteki yerli oranı

    share_asian.append(sum(x.share_asian) / len(x)) # i'nci eyaletteki asyalı oranı

    share_hispanic.append(sum(x.share_hispanic) / len(x)) # i'nci eyaletteki ispanyol kökenlilerin oranı

    

# visualization

f,ax = plt.subplots(figsize=(15,15))

sns.barplot(x=share_white , y=area_list, color='pink', alpha=0.5, label='White American')

sns.barplot(x=share_black , y=area_list, color='blue', alpha=0.5, label='Black American')

sns.barplot(x=share_native_american , y=area_list, color='cyan', alpha=0.5, label='Native American')

sns.barplot(x=share_asian , y=area_list, color='yellow', alpha=0.5, label='Asian American')

sns.barplot(x=share_hispanic , y=area_list, color='red', alpha=0.5, label='Hispanic American')



ax.legend(loc='upper right' , frameon=True) # sağ alt köşede dursun ve frameon arka plan şeridi belli olsun(True)

ax.set(xlabel='Percentage of Races', ylabel='States', title="Percentage of State's Population According to Races")

plt.show()

sorted_data.head()
sorted_data2.head()
# Basic Normalization  (sonucunda iki framede ki değerlerde normalze edilmiş bir şekilde olacak (0-1))

sorted_data.area_poverty_ratio = sorted_data.area_poverty_ratio / max(sorted_data.area_poverty_ratio)

sorted_data2.area_highschool_ratio = sorted_data2.area_highschool_ratio / max(sorted_data2.area_highschool_ratio)

#sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max( sorted_data['area_poverty_ratio'])

#sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max( sorted_data2['area_highschool_ratio'])



#İki data concat() methodu ile birleştiriliyor birleştiriliyor. axis = 1  ==> Yatayda birleştirme işlemi

data = pd.concat([sorted_data,sorted_data2.area_highschool_ratio],axis=1)

data.sort_values('area_poverty_ratio',inplace=True)



#Visualize (Görselleştirme)

f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='area_list',y='area_poverty_ratio',data=data, color='lime',alpha=0.8)

sns.pointplot(x='area_list',y='area_highschool_ratio',data=data, color='red',alpha=0.8)

plt.text(40,0.6,'high school graduate ratio',color='red',fontsize = 17,style = 'italic')

plt.text(40,0.55,'poverty ratio',color='lime',fontsize = 18,style = 'italic')

plt.xlabel('States',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')

plt.grid()

data.head()
import scipy.stats as stats   #library for perasonr

# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# joint kernel density

# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.

# If it is zero, there is no correlation between variables

# Show the joint distribution using kernel density estimation 

# size: plotun büyüklüğü



g = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind="kde", size=7)

g.annotate(stats.pearsonr) # for pearsonr

#g1 = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind="scatter", size=7)

#g2 = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind="hex", size=7)

#g3 = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind="reg", size=7)

#plt.savefig('graph.png')

plt.show()
# you can change parameters of joint plot

# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }

# Different usage of parameters but same plot with previous one

g = sns.jointplot("area_poverty_ratio", "area_highschool_ratio", data=data,size=5, ratio=5, color="r")

g.annotate(stats.pearsonr) # for pearsonr

plt.show()
kill.head()
kill.race.head(10)
kill.race.value_counts()
# matplotlib 

# Öldürülen insanların ırk oranı nedir ?

# datada boş değer varsa temizliyoruz ve kaydediyoruz.

kill.race.dropna(inplace=True)

labels = kill.race.value_counts().index  # w,b,h,a,n,o  (white,black...)

colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0,0,0,0,0]  #pie kendi içinde değerlere bakarak kuracağı ortantıları atayacağı dizi.

sizes = kill.race.value_counts().values  # 1201,618,423,39,31,28



#visualiza ===>      sizes=değerler,  explode =oranların tutulacağı dizi, labels=değer etiketleri,  autpct=değerin virgül sonrası adeti.

plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Killed People According to Races',color = 'black',fontsize = 15)

plt.show()
data.head()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# lmplot 

# Show the results of a linear regression within each dataset

# Diğer plotlardan farklı olarak linear regression'u da vermektedir.

# Dataların en optimum noktasından geçen bir regression result'ı oluşur.

sns.lmplot(x="area_poverty_ratio", y="area_highschool_ratio", data=data)

plt.show()

# y= 1-x

# x=0
data.head()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# cubehelix plot

# shade = (yoğunluğu) dolgunluğu gösteren görselin durumu

# yoğunluğun en fazla olduğu noktada: Dataset'de poverty_ration ve highschool_ratio değerlerinin en fazla tekrarlandığı değerler olduğunu gösterir.

# cut = katmanlardaki kesitlerin kücüklüğünü ifade eder.

sns.kdeplot(data.area_poverty_ratio, data.area_highschool_ratio, shade=True, cut=1)

plt.show()
data.head()
# Show each distribution with both violins and points

# Use cubehelix to get a custom sequential palette

# Tek farklı diğer plotlardan farklı olarak iki niteliğin,kolonun,feature'nin correlation(korelasyon,ilişki)'una bakmaktansa:

# datadaki diğer kolonların,featurelerinin değerlerinin dağılımına bakar.

# pal = görselin rengi,türü,tipini belirleyen kalıptır. Farklı palet tipleride vardır.

# inner = görsel içerisinde yer alan noktaları yani data pointleri gösterir.

# data içerisindeki sadece sayısal değerleri görselleştirir.

# Şekillerin en şişman olduğu kısımlar histogram gibi en çok tekrarlanan değeri ifade eder.

# Bu örnekte area_poverty_ratio featuresinde yaklaşık 0.5 değerinin en çok tekrarlanan değer olduğunu ifade ediyor.

# Area_highschool_ration featuresinde yaklaşık 0.99 değeride en çok tekrarlanan değerdir.

pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=data, palette=pal, inner="points")

plt.show()

data.head()
data.corr()
#correlation map

# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()
kill.head()
kill.gender.unique()
kill.age.unique()
kill.manner_of_death.unique()
# manner of death(olum sekli) : ates edilerek, ates edilerek ve sok tabancasiyla

# gender cinsiyet

# age: yas

# Amaç öldürülen insanların cinsiyetlerini,yaşlarını ve ölüm şekillerini öğrenmek ve görselleştirmek

# Çalışma mantığı x ekseni için "gender" input olarak verilmiştir. boxplot data'nın (kill) içerisine girer ve "gender" kolonundaki unique değerlere erişir.

# hue = classlarına ayırarak işlem yap. (shot - shot and Tasered )

sns.boxplot(x="gender",y="age",hue="manner_of_death",data=kill,palette="PRGn")

plt.show()
kill.head()
# swarm plot

# manner of death(olum sekli) : ates edilerek, ates edilerek ve sok tabancasiyla

# gender cinsiyet

# age: yas

sns.swarmplot(x="gender", y="age",hue="manner_of_death", data=kill)

plt.show()



# Erkekler ve kadınlar için hangi yaşta hangi tipde ölüm gerçekleşmiş rahatça analiz edilebilir.

# sezgisel olarak hangi feature'den sınıflandırmada yardım alacağımızı belirlemede yardımcı olabilir.

# Feature lerin farklı rahatça görmemizi sağlayan bir plottur.

# Dezavantajı, veri sayısı çok fazla ise çizdirmek çok zordur. Fazla veri'de bilgisayar çok zorlanır.
data.head()
# sayısal değerleri plot eder. Scatter ve histogram şeklinde

# pair plot

sns.pairplot(data)

plt.show()
kill.manner_of_death.value_counts()
kill.gender.value_counts()
kill.head()
# kill properties

# Manner of death

sns.countplot(kill.gender)

#sns.countplot(kill.manner_of_death)

plt.title("gender",color = 'blue',fontsize=15)

plt.show()
sns.countplot(kill.manner_of_death)

plt.title("manner_of_death",color = 'blue',fontsize=15)

plt.show()
# kill weapon (öldürmede kullanılan alet)

armed = kill.armed.value_counts()

#print(armed)

plt.figure(figsize=(10,7))

sns.barplot(x=armed[:7].index,y=armed[:7].values) # ilk 7 tipi öldürme aracını aldık

plt.ylabel('Number of Weapon')

plt.xlabel('Weapon Types')

plt.title('Kill weapon',color = 'red',fontsize=15)

plt.show()

# age of killed people

# 25 yaşının üstünde öldürülenler ve altında öldürüleleri görselleştiriyoruz.

# yeni bir frame oluşturuyoruz kill.age kolonu içindeki her bir yaşa bakıyoruz.

# Yaş 25'den büyük veya eşit ise 'above25' valuesi oluşturuyoruz aksi durumda 'below25'



status25 =['above25' if i >= 25 else 'below25' for i in kill.age] #list

df = pd.DataFrame({'age':status25})   #list to dataframe

sns.countplot(x=df.age)

plt.ylabel('Number of Killed People')

plt.title('Age of killed people',color = 'blue',fontsize=15)

plt.show()
# Most dangerous cities

# en tehlikeli şehirler (en tehlikeli 12 şehir)

city = kill.city.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=city[:12].index,y=city[:12].values) # ilk 12 şehir

plt.xticks(rotation=45)

plt.title('Most dangerous cities',color = 'blue',fontsize=15)

plt.show()
# most dangerous states

# en tehlikeli 20 eyalet

state = kill.state.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=state[:20].index,y=state[:20].values)

plt.title('Most dangerous state',color = 'blue',fontsize=15)

plt.show()
# Having mental ilness or not for killed people

# Öldürülen insanların zihinsel rahatsızlığa sahip olup olmaması.

sns.countplot(kill.signs_of_mental_illness)

plt.xlabel('Mental illness')

plt.ylabel('Number of Mental illness')

plt.title('Having mental illness or not',color = 'blue', fontsize = 15)
# Threat types

# Tehdit türleri

sns.countplot(kill.threat_level)

plt.xlabel('Threat Types')

plt.title('Threat types',color = 'blue', fontsize = 15)

plt.show()
kill.head()
# Flee types

sns.countplot(kill.flee)

plt.xlabel('Flee Types')

plt.title('Flee types',color = 'blue', fontsize = 15)

plt.show()
# Having body cameras or not for police

# Polislerde kamera var mıydı yok muydu oranı

sns.countplot(kill.body_camera)

plt.xlabel('Having Body Cameras')

plt.title('Having body cameras or not on Police',color = 'blue',fontsize = 15)

plt.show()
# Kill numbers from states in kill data

sta = kill.state.value_counts().index[:10]

sns.barplot(x=sta,y = kill.state.value_counts().values[:10])

plt.title('Kill Numbers from States',color = 'blue',fontsize=15)

plt.show()