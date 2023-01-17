import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt 

import warnings

warnings.filterwarnings('ignore') 

from collections import Counter



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#5 farklı datasetimiz var ve bunları okuyalım 

median_house_hold_in_come = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv', encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")

percent_over_25_completed_highSchool = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

share_race_city = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv', encoding="windows-1252")

kill = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding="windows-1252")
#2 numaralı soru -Her eyaletin fakirlik oranı

percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()

#poverty_rate değerim string ama bu sekilde isime yaramaz o yüzden float yapmalıyız
percentage_people_below_poverty_level.poverty_rate.value_counts()

#Burada hangi değerden kaç tane olduğunu gördük ve - şeklinde  201 tane veri oldugunu gördük 

#Ve 0 değerinde 1464 tane veri oldugunu gördük peki hangisi fakir değil demek bunu bilmiyoruz

#Burada bunu bilemediğimiz için -değerindekileri 0'a eşitleyip kurtulmamız gerekiyor 

percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace = True) #- değerini 0.0 ile değiştir

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float) #String olan poverty_rate i float yaptık
percentage_people_below_poverty_level.info() #Görüldüğü üzere artık poverty_rate float oldu

area_list=list(percentage_people_below_poverty_level["Geographic Area"].unique())

#ppbpl datamızdaki Geographic area ismindeki columndaki data veri isimlerini görelim

#Burada Area daki state isimlerini olusturduk
#şimdi sıralı şekilde datayı sıralayalım 

area_poverty_ratio = [] #Boş bir liste açtık 

for i in area_list:

    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level["Geographic Area"]==i] #Burada ppbpl datasındaki Geographic Area'ları sırayla aldık 

    #buradaki ==i demek for döngüsündeki her index de veri çekebilmek içindir 

    area_poverty_rate=sum(x.poverty_rate)/len(x) #Burada poverty ratein ortalamasını yani rate i aldık

    area_poverty_ratio.append(area_poverty_rate) #Ve boş listemize ekledik 

data = pd.DataFrame({'area_list': area_list, 'area_poverty_ratio': area_poverty_ratio}) #Yeni bir dataframe olusturduk 

#Sonrada poverrty ratioları büyükten küçüge sıraladık ve index verisini çektik

new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values #Ascending = False azalan sıra demektir 

sorted_data = data.reindex(new_index) #Ve kullancagımız dataya bu indexleri aktardık



#Görsellestirme - BAR PLOT

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'],y=sorted_data['area_poverty_ratio']) #datamızdan x ve y nin verilerini çektik 

plt.xticks(rotation=45) #Burada x eksenindeki veri isimlerini 90 derece yani dikey yazdırmamız içindi

#Eger 45 desem variable isimleri çapraz duracaktı

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States ')

plt.show()
#Şimdi öldürülen en yaygın 15 isim veya soyisimi bulalım 

#Öncelikle datada bir hata var mı diye kontrol edelim Kill dataset kullanıcaz 

kill.head() #Biz burda name parametresini inceleyeceğiz 
kill.name.value_counts()

#Burada TK TK diye bir veri var ki muhtemelen hatalı o yüzden bunu analiz dışı bırakalım 

separate = kill.name[kill.name !='TK TK'].str.split() #burada TK TK ya eşit olmayanları görmek istiyoruz split ise aradaki boşluklar oldugu için 

#yeni bir kelime gibi kabul eder 

a,b = zip(*separate) #Burada unzipledik ve tuble da isim ve soyisimleri toplayacağız 

name_list = a+b #Tuble oldugu için aynı tuble a aldık 

name_count = Counter(name_list)

most_common_names = name_count.most_common(15) #.most_common() bir metoddur ve en yaygınları verir 

x,y = zip(*most_common_names)

x,y = list(x),list(y)



#Görselleştirme - BAR PLOT 

plt.figure(figsize =(15,10)) 

sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x))) 

#Burada palette = ..... metodu x uzunluğu kadar birbiri ile alakalı farklı renkte gösterim yapılmasını sağlar 

plt.xlabel("Name or Surname of killed people")

plt.ylabel("Frequency")

plt.title("Most Common 15 Name or Surname of killed people")
#Şimdi 25 yaş altında lise mezunu olanların oranını inceleyelim 

percent_over_25_completed_highSchool.head()
percent_over_25_completed_highSchool.info()
#yine eksik data var mı bakalım - Yüzdeyi istediğimiz için percent_completed_hs yi inceleriz 

percent_over_25_completed_highSchool.percent_completed_hs.value_counts() 

percent_over_25_completed_highSchool.percent_completed_hs.replace(["-"],0.0,inplace=True)

#Ayrıca bu veri string(object) oldugu için float yapalım 

#percent_over_25_completed_highSchool=float(percent_over_25_completed_highSchool.percent_completed_hs) #şeklinde hata verir o yüzden astype kullanırız .

percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)



#şimdi bu durumdaki geographic area ları görelim 

area_list = list(percent_over_25_completed_highSchool["Geographic Area"].unique())

area_highschool = []

for i in area_list:

    x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool["Geographic Area"]==i]

    area_highschool_rate = sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)

    

# Sıralayalım 

data=pd.DataFrame({'area_list':area_list, 'area_highschool_ratio': area_highschool})

new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values

sorted_data2=data.reindex(new_index)



#Görselleştirme Bar Plot 

plt.figure(figsize=(18,15))

sns.barplot(x=sorted_data2['area_list'],y=sorted_data2['area_highschool_ratio'])

plt.xticks(rotation = 45)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Gives States' Population Above 25 that has graduated High School")

plt.show()          
#YATAY BARPLOT 

#Eyaletlerdeki ırkların oranını görselleştirelim SHARE RACE CITY DF İLE 

share_race_city.head()

share_race_city.info()
#Bu oranları object değil sayı (float) yapmalıyız 

#Ama önce datalar içinde bir hatalı giris var mı bakalım ornegin share_white ile bakalım

share_race_city.share_white.value_counts() 

#Görünürde birşey yok ama önceki datasetlerden gördüğümüz kadarıyla X ve - vardı onları 0 yapalım 

share_race_city.replace(["-"],0.0,inplace=True)

share_race_city.replace(["(X)"],0.0,inplace=True)

share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_asian','share_hispanic']]=share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_asian','share_hispanic']].astype(float)

#Burada city ve G.area dışındaki verileri float yaptık

area_list = list(share_race_city["Geographic area"].unique())

share_white = []

share_black = []

share_native_american = []

share_asian = []

share_hispanic = []

for i in area_list:

    x = share_race_city[share_race_city['Geographic area']==i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black) / len(x))

    share_native_american.append(sum(x.share_native_american) / len(x))

    share_asian.append(sum(x.share_asian) / len(x))

    share_hispanic.append(sum(x.share_hispanic) / len(x))

#Görselleştirme 

f,ax = plt.subplots(figsize=(9,15))

sns.barplot(x=share_white, y=area_list, color="green",alpha =0.5, label="White")

sns.barplot(x=share_black,y=area_list,color='blue',alpha = 0.7,label='African American')

sns.barplot(x=share_native_american,y=area_list,color='cyan',alpha = 0.6,label='Native American')

sns.barplot(x=share_asian,y=area_list,color='yellow',alpha = 0.6,label='Asian')

sns.barplot(x=share_hispanic,y=area_list,color='red',alpha = 0.6,label='Hispanic')   



ax.legend(loc='lower right',frameon = True) # Alt taraf sağ köşede frameon=True yaparsam bir kutu içinde labelleri gösterir 

ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races")

plt.show()
#POINT PLOT 

# High school graduation rate vs Poverty rate of each state - Mezuniyet vs Fakirlik oranı

print(sorted_data.head())

print(sorted_data2.head())
#Hali hazırda olusturdugumuz sorted_data ve sorted_data2 kullanarak görselleştirmemizi yapalım 

#Ama burada veriler görüldüğü gibi biri 25 lerde iken diğer data 75 lerde o yüzden max a bölerek normalize ettik

sorted_data["area_poverty_ratio"]=sorted_data["area_poverty_ratio"]/max(sorted_data["area_poverty_ratio"])

sorted_data2["area_highschool_ratio"]=sorted_data2["area_highschool_ratio"]/max(sorted_data2["area_highschool_ratio"])

#Şimdi datalarımızı concat komutu ile birleştirelim 

data=pd.concat([sorted_data,sorted_data2["area_highschool_ratio"]],axis=1)

data.sort_values("area_poverty_ratio",inplace=True) 

#Görselleştirme 

f, ax1=plt.subplots(figsize=(20,10))

sns.pointplot(x="area_list",y="area_poverty_ratio",data=data,color="lime",alpha=0.75) #x ve y ye string atabilmemizi sağlayan şey data=data yapısıdır 

#soldaki data kalıp = data ise bizim datamıza verdiğimiz isim, bu sayede columnları string seklinde yazarak atama yapabilirz 

sns.pointplot(x="area_list",y="area_highschool_ratio",data=data,color="red",alpha=0.8)

plt.text(40,0.6,'high school graduate ratio',color='red',fontsize = 17,style = 'italic')

plt.text(40,0.55,'poverty ratio',color='lime',fontsize = 18,style = 'italic') #Burada 40,0.55 olan kısım kordinat ve tabloya string yazıyoruz

plt.xlabel('States',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')

plt.grid()
#JOINT PLOT 

# High school graduation rate vs Poverty rate of each state - Mezuniyet vs Fakirlik oranı

data.head()
#Bu türde tek komut ile yazdırabiliyoruz ve size plotun boyutudur

g=sns.jointplot(data.area_poverty_ratio,data.area_highschool_ratio,kind="kde",size=7)

plt.savefig("graph.png")

plt.show()
# you can change parameters of joint plot

# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }

# Different usage of parameters but same plot with previous one

g = sns.jointplot("area_poverty_ratio", "area_highschool_ratio", data=data,size=5, ratio=3, color="r")

#scatter type default olarak geldiği için yazmadık 
#PIE CHART 

#Race rates according to kill data 

kill.race.head(15)
kill.race.dropna(inplace = True) #race column da boş varsa attık

kill.race.value_counts() #Hangi ırktan kac tane oldugunu gördük 
labels = kill.race.value_counts().index

colors=["grey","blue","red","yellow","green","brown"]

explode=[0,0,0,0,0,0]

sizes = kill.race.value_counts().values



#Görsellestirme 

plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%') #bir ondalıklı göstermek için autopct ekledik

plt.title("Killed People According to Races",color ="blue", fontsize =15,fontstyle ="italic")

plt.show()
#LM PLOT 

#High school graduation rate vs Poverty rate of each state with different style of seaborn code

# Show the results of a linear regression within each dataset

sns.lmplot(x="area_poverty_ratio",y="area_highschool_ratio", data=data)

plt.show()
#KDE PLOT (KERNEL DENSITY ESTIMATION PLOT)

# high school graduation rate vs Poverty rate of each state with different style of seaborn code

sns.kdeplot(data.area_poverty_ratio,data.area_highschool_ratio,shade=5,cut=5) #cut büyüklüğü ifade eder ve değer küçüldükce grafik büyür

plt.show()
#Vıolin Plot 

# Show each distribution with both violins and points

#Bu tarz ile farklı featureların değer dagılımına bakarız, korelasyon yerine 

print(data.head())

pal=sns.cubehelix_palette(2,rot=-.5,dark=.3) #Bu renk kodudur ve googledan seaborn palette yazılarak bircok tür bulunabilir 

sns.violinplot(data=data, palette=pal, inner="points") #points komutu ile noktalı bicimde görmemizi saglar ve noktaların oldugu yerler degerlerin oldugu yerlerdir 

plt.show() #Sadece numerik seyleri görsellestiren bir türdür 

#HEATMAP - Korelasyon Haritası

#high school graduation rate vs Poverty rate of each state with different style of seaborn code

print(data.corr()) #Görüldüğü üzere -0.8 civarı yani ters orantılıdır. 

f,ax =plt.subplots(figsize=(5,5))

sns.heatmap(data.corr(),annot=True, linecolor= "red",linewidths = .5, fmt =".1f",ax=ax) #annot = True sayesinde kutuların içinde de değerleri verir

plt.show()

#BOX PLOT 

#Manner of Death 

print(kill.head())

#Cinsiyet ve yasını da görmek istiyoruz 

sns.boxplot(x="gender", y="age",hue="manner_of_death",data=kill,palette="PRGn") #hue = class demektir yani classlarına ayırmak demek, shot ve shot and Teasered olarak ayırdı

plt.show()
#SWARM PLOT 

#Manner of Death 

#Cok kullanıslıdır ama data sayısı çok ise dezavantajlıdır.

sns.swarmplot(x="gender",y="age",hue = "manner_of_death", data=kill)

plt.show()
#PAIR PLOT 

#Fakirlik vs lise mezuniyeti 

data.head()
sns.pairplot(data)

plt.show()
#COUNT PLOT 

#.value_counts() metodunun grafikleştirilmesi gibi düşünülebilir.

kill.manner_of_death.value_counts()
sns.countplot(kill.gender)

# sns.countplot(kill.manner_of_death)

plt.title("gender",color="red",fontsize=15)

plt.show()

#kill weapon 

armed = kill.armed.value_counts()

print(armed)

plt.figure(figsize = (10,7))

sns.barplot(x=armed[:7].index, y=armed[:7].values)

plt.ylabel("Number of weapon")

plt.xlabel("Weapon Types")

plt.title("Kill Weapon",color="blue",fontsize=20,fontstyle="italic")
#Age of Killed People 

above25=["above25" if i >= 25 else "below25" for i in kill.age]

df = pd.DataFrame({"age":above25})

sns.countplot(x=df.age)

plt.ylabel("Number of Killed People")

plt.xlabel("Age of Killed People",color ="red",fontsize = 15)

plt.show()
# Race of killed people

sns.countplot(data=kill, x='race')

plt.title('Race of killed people',color = 'blue',fontsize=15)

plt.show()
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