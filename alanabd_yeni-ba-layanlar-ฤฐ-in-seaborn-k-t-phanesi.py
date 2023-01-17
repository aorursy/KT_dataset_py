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
# Veri okuma
median_house_hold_in_come = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding="windows-1252")
percentage_people_below_poverty_level = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")
percent_over_25_completed_highSchool = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")
share_race_city = pd.read_csv('../input/ShareRaceByCity.csv', encoding="windows-1252")
kill = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")
#Her eyaletteki yoksulluk oranı 
percentage_people_below_poverty_level.info()
# verimizi incelediğimiz zaman içerisinde 201 adet "-" ifadesinin bulunduğu görülmektedir. Burada veri önişlem aşamasında bu değerlerin olduğu bölümlerde eksik veri
#çalışması yapmamız gerekmektedir. Bunun için ya bu değeri içeren satırlara görmezden gelme, veriden silme, oralama değerle değiştirme, rasgele bir değer girme
# gibi seçeneklerden birisini uygulayacağız. Biz burada 0 değerini girerek burada "-" ifadesi ile 0 değeri girildiğini varsayacağız.
percentage_people_below_poverty_level.poverty_rate.value_counts()
#aşağıdaki işlem ile verimizde yer alan "-"" ifadelerinin 0.0 ile değişimi ve ardından inplaca=True ifadesi ile de bu yapılan işlemin verimizde de geçerli olmasını sağladık
percentage_people_below_poverty_level.poverty_rate.replace(["-"],0.0, inplace=True)

#yaptığımız işlemin sonucunun görmek için 
percentage_people_below_poverty_level.poverty_rate.value_counts()
#gördüğünüz gibi artık "-" değerlerinin yerinde 0.0 değerinin olduğu görülmektedir. Fakat verimizde iki farklı sıfır varmış gibi bir durum olduğu gözlemlenmektedir.
#bunun nedeni ise yukarıdaki satırlarda verimiz hakkında bilgi almak için yazdığımız info() metedu ile anlaşılabilmektedir. poverty_rate alanındaki değerlerin object 
#olduğu görülmektedir. Yoksulluk oranı ifadesi sayısal bir değer içeren anlamına geldiği için ise yapılması gereken işlem bu sütundaki değerlerin sayısal değere dönüştürülmesidir.

percentage_people_below_poverty_level.poverty_rate=percentage_people_below_poverty_level.poverty_rate.astype(float)
percentage_people_below_poverty_level.poverty_rate.value_counts()
# görüldüğü gibi önceki sonuçlarda 1464 ve 201 adet olmak üzere iki farklı sıfır değeri varken dönüşüm işleminden sonra ikisinin toplamı kadar sıfır olduğu görülmektedir.
area_list=list(percentage_people_below_poverty_level["Geographic Area"].unique())
area_list
"""
Burada yapılan işlem şudur:
    Öncelikle for döngüsü yardımıyla yukarıda oluşturduğumuz area_list isimli listemiz içerisinde dolaşıyoruz
    Daha sonra for döngüsünün içerisinde yer alan x değişkeni filtreleme sonucu ile elde edilen yeni verimizi elde ediyoruz. Misal ilk değer olan "AL" değeri için;
    verimizde bu değere sahip olan bütün veriler x değişkeninde saklanmaktır. Daha sonra ise area_poverty_rate ile poverty_rate değerlerinin ortalamasını alıyoruz. 
    daha sonra ise for döngüsünün üzerinde oluşturduğumuz eyalet bazlı olarak poverty_rate değerlerinin ortalamasını saklayacağımız area_poverty_ratio listesinin 
    içerisine atıyoruz.
"""
area_poverty_ratio=[]
for i in area_list:
    x=percentage_people_below_poverty_level[percentage_people_below_poverty_level["Geographic Area"]==i]
    area_poverty_rate=sum(x.poverty_rate)/len(x)
    area_poverty_ratio.append(area_poverty_rate)
area_poverty_ratio
"""
    data verisini oluştururken dictionary kullanışmıştır. öncelikle data oluşturulurken sütun isimleri ve alacağı değerler belirlenmiştir.
    Burada görüldüğü gibi sütun isimleri area_list ve area_poverty_ratio, alacağı değerlerde bu isimlerde yukarıda oluşturduğumuz listelerdir.
    daha sonra pandas kütüphanesinin DataFrame() metodu yardımıyla oluşturulan dictionary dataframe'e dönüştürülmüştür. 
    new_index isimli dizide ise data isimli 2 sütunlu verimizin area_poverty_ratio sütununa göre sıralanmış halinin index değerleri saklanmaktadır.
    daha sonra sorted_data isimli yeni veri setimizde verimizin index'inde değişiklik yaptığımızı ve bu yeni index'in new_index isimli dizide saklanan
    verimizin area_poverty_ratio göre büyükten küçüğe sıralanmış haline ait olan index numaraları bulunmaktad. Yani daha önceden data içerisinde 25.
    sırada olan MS eyaleti ve ona ait olan veri artık ilk sırada bulunmakta. Nedeni tabiki area_poverty_ratio değeri en yüksek olan değer bu eyalete ait olduğu için
    
    
    
"""
data = pd.DataFrame({'area_list': area_list,'area_poverty_ratio':area_poverty_ratio})
new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)
"""
GÖRSELLEŞTİRME
plt.figure(figsize=(15,10))==> ile grafiğimizin boyutunu belirliyoruz. 
sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])===> oluşturacağımız grafiğin seaborn kütüphanesinde yer alan barplot olduğu ve
                                                                                grafik çizdirilirken x ve y eksenindeki değerleri nereden alacağını belirtiyoruz.
plt.xticks(rotation= 45)===> Barplot da yer alan sütunların hangi eyalete ait olduğunu yani x eksenindeki verilerin neye ait olduğunu gösteren ifadelerin hangi
                            açıyla yazılacağını belirtiyoruz.

plt.xlabel('Eyaletler')===> x ekseninde yazılacak ifade
plt.ylabel('Suç oranları')====>y ekseninde yazılacak ifade
plt.title('Eyaletlerin ortalama suç oranları')====>Grafiğimizin başlığı
    
"""

plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])
plt.xticks(rotation= 45)
plt.xlabel('Eyaletler')
plt.ylabel('Suç oranları')
plt.title('Eyaletlerin ortalama suç oranları')
kill.name
"""
En sık öldürülen isim ve soyismi bulalım bunun için öldürülen kişilerin isimlerinin saklandığı PoliceKillingsUS verisetini dataframe olarak eklediğimiz kill dataframe'ine
bakacağız
Aşağıdaki işlem ile kill.name isimli sütunda yer alan ve değeri 'TK TK' olmayan verileri ['Matthew','Folden'] haline deönüştüreceğiz(önceki gösterim şekli Matthew Folden 
şeklinde yan yana arasında boşluk olacak şekilde)
"""
separate = kill.name[kill.name != 'TK TK'].str.split() 

"""
Burada ise unzip işlemini gerçekleştiriyoruz. Yani separate verimizin içerisinde yer alan iki sütundaki değerlerden ilk sütunu
a değişkenine, ikinci sütunu ise b değişkenine atıyoruz.
"""
a,b = zip(*separate)
"""
amacımız tüm isim ve soyisim listesi içerisinde en fazla yer alan değerleri bulmak olduğu için iki listemizi birleştirip tek boyutlu isim ve soyisimlerinden oluşan yeni
listemizi oluşturuyoruz.
"""
name_list = a+b
"""
Counter metodu yardımıyla her bir değerden kaç adet olduğunu bulacağız. Aşağıdaki işlemin sonucunda her bir değer için 
'Tim':3 gibi değerler oluşacak. Bu Tim değerinin bizim listemizde 3 adet bulunduğunu göstermektedir.
"""
name_count = Counter(name_list)
most_common_names = name_count.most_common(15) #most_common metordu listedeki en fazla bulunan kaç değeri alacağımızı belirtmek için kullanacağız. 
x,y = zip(*most_common_names) # x değişkenine isim veya soyisim değerlerinin olduğu bölümü, y değişkenine ise sayısını atıyoruz
x,y = list(x),list(y)#x ve y'yi listeye dönüştürüyoruz
# 
plt.figure(figsize=(15,10))
ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))
plt.xlabel('Öldürülen kişilere ait isim ve soyisimler')
plt.ylabel('Frekans')
plt.title('En sık öldürülme vakasının yaşandığı 15 isim veya soyisim')
plt.show()
"""
Şimdi ise cevap arayacağımız soru "Eyaletlerdeki 25 yaş üstü kişilerin yüksek okul mezun olma oranı"
Bunun için kullanacğımız veri seti PercentOver25CompletedHighSchool ve bu veri setini kullandığımız dataframe ise percent_over_25_completed_highSchool 
Öncelik bu dataframe'i inceleyerek işe başlayalım.
"""
percent_over_25_completed_highSchool.head()
percent_over_25_completed_highSchool.info()
"""
Burada ki en önemli sorunlardan birisi percent_completed_hs değerlerinin sayısal olmasını gerekirken string olması
"""
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()
"""
Aşağıda görüldüğü gibi 197 adet verinin değeri girilmemiş. Girilmeyen değerleri ya yoksayacağız ya da daha öncede uyguladığımız gibi 0 değerini vereceğiz. Daha sonra 
ise bu sütunda yer alan değerleri sayısal değere dönüştüreceğiz.
"""
percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0,inplace = True)
percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()

#Aşağıda görüldüğü gibi daha önce - yazan bölümde artık 0.0 değeri bulunmakta

"""
Her eyaletteki yoksulluk oranı bölümündekine benzer bir işlem ile görselleştirme işlemi yapılmıştır. Tek fark orada barplot büyükten küçüğe sıralama 
için kullanılımış burada ise sıralama küçükten büyüğe şeklinde seçilmiştir. 
"""
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
# visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['area_highschool_ratio'])
plt.xticks(rotation= 90)
plt.xlabel('Eyaletler')
plt.ylabel('Lise mezuniyet oranı')
plt.title("Her bir eyaletteki 25 üstü kişilerin liseden mezun olma oranı")
plt.show()
"""
Eyaletlerdeki farklı kökenlerdeki(siyah,beyaz,latin,kızılderili,asyalı) insanların yüzdeleri
Bu soruya cevap verebilmek için eyaletlerdeki şehirlerde bulunan ırklara ait değerlerin hangi veri setinde olduğuna bakmamız gerekir.
share_race_city isimli dataframe'imiz bizim ulaşmak istediğimiz verilere ait verisetini sakladığımız dataframe'imiz.
Aldığı değerlere ait bilgi almak için info(), head() vb. metodları kullanacağız.
"""
share_race_city.head()
share_race_city.replace(['-'],0.0,inplace = True)#daha önce de yaptığımız gibi burada yapılan share_race_city isimli daraframe'imizde bulunan eksik veri sorununa çözüm üretmek
share_race_city.replace(['(X)'],0.0,inplace = True) #bu veri setimizde iki farklı değer girilmiş bilinmeyen değer olarak bunlar "-" ve "X"
#aşağıda ise sayısal değer içeren bölümlerde string değer olduğu için bu değerleri sayısal değere dönüştürüyoruz astype() metodu ile
share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)
#for döngüsünde yapılan işlem de dahil olmak üzere yapılan işlemler 2. soruda verilen yöntemle aynı işlemlerdir sadece birden fazla değişkende işlem yapılmaktadır
area_list = list(share_race_city['Geographic area'].unique())
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

# Görselleştirme işlemi için bar plot kullanılmıştır
f,ax = plt.subplots(figsize = (9,15))
sns.barplot(x=share_white,y=area_list,color='green',alpha = 0.5,label='Beyaz' )
sns.barplot(x=share_black,y=area_list,color='blue',alpha = 0.7,label='Afrika Kökenli')
sns.barplot(x=share_native_american,y=area_list,color='cyan',alpha = 0.6,label='Yerli Amerikalı')
sns.barplot(x=share_asian,y=area_list,color='yellow',alpha = 0.6,label='Asyalı')
sns.barplot(x=share_hispanic,y=area_list,color='red',alpha = 0.6,label='Esmer')

ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu
ax.set(xlabel='Irkların eyaletlere göre yüzdesi', ylabel='Eyaletler',title = "Irklara göre Amerikan nüfusunun yüzdesi ")
plt.show()
sorted_data
#ilk öncelikli olarak daha önceden kullandığımız sorted_data verisini kullanarak eyaletlerdeki ortalama yoksulluk oranını buluyoruz
sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max( sorted_data['area_poverty_ratio'])
sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max( sorted_data2['area_highschool_ratio'])
data = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)
data.sort_values('area_poverty_ratio',inplace=True)
#Görselleştirme sürecinde point plot kullanacağız. burada 2 farklı değeri tek bir grafik arayüzünde görselleştireceğimiz için sub plotda kullanacağız ekstradan
#pointplot'un aldığı değerlerden x, x eksenindeki değeri, y, y ekseninde yer alan değeri, data ise verimizi aldığımız verisetimizi ifade etmektedir.
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color='lime',alpha=0.8)
sns.pointplot(x='area_list',y='area_highschool_ratio',data=data,color='red',alpha=0.8)
plt.text(40,0.6,'Lise mezun oranı',color='red',fontsize = 17,style = 'italic')
plt.text(40,0.55,'Yoksulluk oranı',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('Eyaletler',fontsize = 15,color='blue')
plt.ylabel('oranlar',fontsize = 15,color='blue')
plt.title('lise mezuniyet ve yoksulluk oranı',fontsize = 20,color='blue')
plt.grid()
g = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind="kde", size=7)

plt.show()
g = sns.jointplot("area_poverty_ratio", "area_highschool_ratio", data=data,size=5, ratio=3, color="r")
kill.head()
kill.race.value_counts()
kill.race.head()
#burada yaptığımız eğer kill datasının race sütununda değer içermeyen bir satır varsa onu sil ve inplace=True ile de değiştirilen durumu kill.race'e ata anlamına gelir
kill.race.dropna(inplace = True)
labels = kill.race.value_counts().index # kill.race.value_counts()'ın sonucu çıkan değerlerden değerleri index, değerlerin sayısını value olarak labels'ın içerisine atacak
colors = ['grey','blue','red','yellow','green','brown']
explode = [0,0,0,0,0,0]
sizes = kill.race.value_counts().values

# visual
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Irklara göre ölüm oranları',color = 'blue',fontsize = 15)
plt.show()
data.head()
sns.lmplot(x="area_poverty_ratio",y="area_highschool_ratio",data=data)
plt.show()
#aşağıdaki plot uygulamasında birinci bölüm x ekseni, ikinci bölüm y eksenini belirtmektedir. 
#shade bölümü ise grafik oluştururken renklendirme mi yoksa sadece çizgilerden mi oluşacağını göstermektedir. Aşağıda False değer almış hali de çizdirilmiştir.
# son bölüm cut ise grafiğimizin daha geniş bir şekilde temsil edilmesi amacıyla kullanılmaktadır.
sns.kdeplot(data.area_poverty_ratio,data.area_highschool_ratio,shade=True,cut=5)
plt.show()
sns.kdeplot(data.area_poverty_ratio,data.area_highschool_ratio,shade=False,cut=5)
plt.show()
pal=sns.cubehelix_palette(2,rot=.5,dark=.3)# bu bölüm görselleştirmede kullanacağımız paletteki renkleri belirlemek için kullanacağız
sns.violinplot(data=data,palette=pal,inner="points")
plt.show()
