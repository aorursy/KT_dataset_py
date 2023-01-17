import seaborn as sns #seaborn kütüphanemi dahil ettim.

import numpy as np #numpy kütüphanemi dahil ettim.

import pandas as pd #pandas kütüphanemi dahil ettim.

from matplotlib import pyplot as plt #matplotlib kütüphanemi dahil ettim.

tablet = pd.read_csv("../input/tabletcsv/tablet.csv",sep=",") #Verilen csv dosyasında veriler virgülle ayrıldıkları için sep'i virgül olarak seçtim.

df=tablet.copy() #İşlem yaparken asıl dosyamın etkilenmemesi için kopyasını df değişkenime atadım.
df.head() #Verimizde ilk 5 elemanı görüntülüyorum.
df.shape #Verimiz 20 Öznitelik 2000 adet gözlemden oluşmaktadır.
df.info()
df.describe().T #En büyük standart sapma RAM'dedir.Bu şu demektir tabletlerin RAM'leri arasında fark fazladır.

#En küçük standart sapma Kalinlik'tadır.Bu şu demektir tabletlerin Kalinlik'lari arasında çok az bir fark vardır.Yani neredeyse tüm tabletlerin klaınlıkları birbiriyle aynıdır.
df.groupby(["FiyatAraligi"]).std()
df.groupby(["FiyatAraligi"]).mean()
df.isnull().sum() #OnKameraMP özniteliğimde 5 adet ve RAM özniteliğimde ise 12 adet eksik veri bulunmamaktadır.
df.dtypes #Verilen özniteliklerin Data tiplerini görüntülüyorum.
sns.violinplot(x=df["CekirdekSayisi"],color="pink"); #Dağılım,normal dağılım değildir.
sns.violinplot(df["FiyatAraligi"],df["RAM"]) #Pahalı tabletlerin RAM'leri 2000 ile 4000,normal fiyatlı tabletlerin 1000 ile 4000,ucuz tabletlerin 0 ile 3000 ve çok ucuz tabletlerin ise 0 ile 2000 arasındadır. 
sns.scatterplot(df["Agirlik"],df["BataryaGucu"]) #Aralarında herhangi bir ilişki bulunmamaktadır.
sns.violinplot(x= "WiFi" , y="Agirlik" , data=df); #WiFi'nin olup olmaması ağırlığı görüldüğü üzere etkilememektedir.
sns.scatterplot(x= "WiFi" , y="Agirlik" , data=df);
sns.violinplot(x = "CiftHat" , y ="OnKameraMP" , data = df);#Çift hatın olup olmaması ile önKameraMP arasında bir ilişki yoktur fakat tablodan OnKameraMP'si arttıkça tabletlerin azaldığını net bir şekilde görebiliriz.Yani OnKameraMP'si yüksek olan tablet sayısı azdır.
sns.scatterplot(x = "CiftHat" , y ="OnKameraMP" , data = df);
sns.countplot(df["Renk"]) #Renklerin kaçar adet dağıldıklarını bu tabloda görebiliyoruz.
sns.violinplot(x = "4G" , y ="CekirdekSayisi" , data = df);#4G'si olan tabletlerde çekirdek sayısı 4 ile 6 aralığındayken tablet sayısı azalmıştır.
sns.violinplot(x = "3G" , y ="DahiliBellek" , data = df);#Aralarında bir ilişki yoktur.
sns.violinplot(x = "Dokunmatik" , y ="CozunurlukGenislik" , data = df);#Aralarında bir ilişki yoktur.
sns.violinplot(x = "Bluetooth" , y ="CozunurlukYükseklik" , data = df);#Aralarında bir ilişki yoktur.
sns.scatterplot(x = "BataryaOmru", y = "BataryaGucu", hue = df["Dokunmatik"], data = df);#Ayırt edilmesi çok zordur.Herhangi bir kümeleme yapamayız.
sns.violinplot(df["FiyatAraligi"],df["MikroislemciHizi"])#Mikro işlemci hızının fiyat aralığı ile bir ilişkisi yoktur.Çünkü her fiyat türünde(çok ucuz,ucuz,normal,pahalı gibi) mikro işlemci hızı yüksek olan tablet bulunmaktadır.
sns.violinplot(x=df["MikroislemciHizi"]); #Dağılım,normal dağılım değildir.
sns.distplot(df["MikroislemciHizi"],bins=16, color="purple");
sns.jointplot(x = "CozunurlukYükseklik", y = "DahiliBellek", data = df); #Aralarında bir ilişki yoktur.Dahili belleğin frekansının en yüksek olduğu aralık 0 ile 250,ÇözünürlükYukseklik'in en yüksek frekansa sahip olduğu aralık ise 10 ile 20'dir.
sns.violinplot(df["FiyatAraligi"],df["DahiliBellek"])
sns.violinplot(df["DahiliBellek"])#Dağılım normal dağılım değildir.
sns.distplot(df["DahiliBellek"],bins=16, color="red");
sns.violinplot(x = "Kalinlik", y = "BataryaOmru", data = df); #Görüldüğü gibi batarya ömrü ve kalınlık özelliklerinin bir ilişkisi bulunmamaktadır.
sns.jointplot(x = "Kalinlik", y = "BataryaOmru", data = df)#Tablet kalınlığı frekansı en çok 0 ile 0.2 aralığındayken BataryaOmru'nun ise 17.5 ile 20 arasındadır.
sns.violinplot(df["BataryaGucu"],df["FiyatAraligi"])#Batarya gücü fazla olan tablet sayısı fiyat aralıklarına göre en çok pahalı tabletlerdeyken en az ise çok ucuz tabletlerdedir.
sns.scatterplot(x = "ArkaKameraMP", y = "Kalinlik", hue = df["FiyatAraligi"], data = df);#Ayırt edilmeleri çok zordur.Çünkü ArkaKameraMP ve Kalinlik'i yüksek veya düşük olup fiyatı değişken olan ürünler var.
df.cov()

#Batarya gücü ile DahiliBellek,CekirdekSayisi,CozunurlukGenislik,RAM arasında negatif MikroislemciHizi,OnKameraMP,Kalinlik,Agirlik,ArkaKameraMP,CozunurlukYükseklik,BataryaOmru pozitif bir ilişki bulunmaktadır.

#Ön kamera MP ile MikroislemciHizi,DahiliBellek,Kalinlik,CekirdekSayisi,CozunurlukYükseklik,CozunurlukGenislik,BataryaOmru ile negatif BataryaGucu,Agirlik,ArkaKameraMP,RAM arasında pozitif bir ilişki bulunmaktadır.
corr=df.corr()

corr #Buradaki en güçlü pozitif yönlü ilişki OnKameraMP ve ArkaKameraMP arasındadır.Korelasyonları 0.645697'tür.Bu durumda korelasyon 1'e yaklaştıkça mükemmelleşir.
df.plot(x='OnKameraMP', y='ArkaKameraMP', style='*')
corr=df.corr() #Isı haritasında da OnKameraMP ve ArkaKameraMP arasında güçlü,CozunurlukGenislik ve CozunurlukYukseklik arasında ise zayıf bir ilişki olduğunu görüyoruz.

sns.heatmap(corr,

           xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
sns.jointplot(x=df["OnKameraMP"],y=df["ArkaKameraMP"],kind="reg");#Korelasyon ilişkisinin en güçlü olduğu ön ve arka kamera verilerini görselleştirdim.
sns.scatterplot(x = "OnKameraMP", y = "ArkaKameraMP", data = df);
sns.jointplot(x = "OnKameraMP", y = "ArkaKameraMP", data = df);#OnKameraMP ve ArkaKameraMP frekansının yüksek olduğu aralık 0 ile 2.5 arasındadır.
sns.jointplot(x=df["OnKameraMP"],y=df["ArkaKameraMP"],kind="kde")
sns.lmplot(x="OnKameraMP",y="ArkaKameraMP", data = df);#Pozitif yönde güçlü bir ilişki vardır.
df.corr()["OnKameraMP"]["ArkaKameraMP"] #Korelasyon katsayısı 1'e yaklaştıkça mükemmelleşir.
sns.scatterplot(x = "OnKameraMP", y = "ArkaKameraMP", hue = df["FiyatAraligi"], data = df);#Buradan arka ve ön kamera pikselleri arttıkça fiyatın attığını kesin olarak söyleyemeyiz.Ancak iyi bir arka ve ön kameraya sahip fiyatı uygun olan telefonlarda bulunsa da bu kamera değişkenleri ve fiyat aralığı bizim için ayırt edici özellik olabilir.
df.FiyatAraligi.value_counts() #Görüldüğü üzere hedef değişken dengeli dağılmıştır.
sns.violinplot(x=df["OnKameraMP"]) #OnKameraMP değişkeni normal dağılıma sahip değildir.
sns.violinplot(x=df["ArkaKameraMP"])#Aynı şekilde ArkaKameraMP değişkenide normal dağılıma sahip değildir.
sns.distplot(df["OnKameraMP"],bins=16, color="red");
sns.distplot(df["ArkaKameraMP"],bins=16, color="red");
sns.violinplot(df["FiyatAraligi"],df["OnKameraMP"])
sns.violinplot(df["FiyatAraligi"],df["ArkaKameraMP"])
sns.countplot(df["FiyatAraligi"]) #Görselde 500 adet normal,500 adet pahalı,500 adet ucuz ve 500 adet çok ucuz tablet bulunmaktadır.Bu tablo hedef değişkenimiz olan FiyatAraligi'nın dengeli dağıldığını doğrulamaktadır.
sns.jointplot(x=df["CozunurlukYükseklik"],y=df["CozunurlukGenislik"],kind="reg");#Korelasyon ilişkisinin zayıf olduğu çözünürlük uzunluk ve genişlik veirlerini görselleştirdim.
sns.scatterplot(x=df["CozunurlukYükseklik"],y=df["CozunurlukGenislik"]);
sns.scatterplot(x = "CozunurlukYükseklik", y = "CozunurlukGenislik", hue = df["FiyatAraligi"], data = df)#Bu verilere bakarak fiyat aralığı karşılaştırması yapmak zordur.Çünkü aynı özelliklere sahip fakat fiyat farkları oluşan ürünler bulunmaktadır.Bundan dolayı ben ön ve arka kamera bilgileri ile fiyat aralığını kullanacağım.
sns.violinplot(x=df["CozunurlukYükseklik"]) #Normal dağılıma sahip değildir.
sns.violinplot(x=df["CozunurlukGenislik"]) #Normal dağılıma sahip değildir.
df.hist(figsize =(20,20), color = "purple")

plt.show()
df.isnull().sum()#OnKameraMP'de 5 adet , RAM değişkenimde ise 12 adet eksik verim bulunmaktadır.
df.isnull().sum().sum() #Toplamda 17 adet eksik veri bulunmaktadır.
df["OnKameraMP"].unique()
df["RAM"].unique()
import missingno
missingno.matrix(df,figsize=(15,5));#Eksik değer barındıran 2 özniteliğim var.

#"Bu 2 öznitelikte eksik değerler rassal şekilde mi meydana geldi yoksa bu değişkenlerin eksik değerleri arasında bir ilişki var mı?" sorusunu görselleştiriyoruz.
missingno.heatmap(df, figsize= (15,8));
def eksik_deger_tablosu(df): 

    eksik_deger = df.isnull().sum()

    eksik_deger_yuzde = 100 * df.isnull().sum()/len(df)

    eksik_deger_tablo = pd.concat([eksik_deger, eksik_deger_yuzde], axis=1)

    eksik_deger_tablo_son = eksik_deger_tablo.rename(columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})

    return eksik_deger_tablo_son
eksik_deger_tablosu(df)#Bu iki dğeişkenimde de eksik olan gözlemler çok azdır.

#ArkaKameraMP'de bu oran %0.25 iken RAM'de ise %0.60'dır.
df[df["OnKameraMP"].isnull()]#Çoğunlukla OnKameraMP bilgisi olmayan tabletlerin Bluetooth özelliğide bulunmamaktadır.
sns.countplot(df[df["OnKameraMP"].isnull()]["Bluetooth"]);

# Eksik OnKameraMP değerine sahip tabletlerin Bluetooth frekanslarını görüntüledik.
sns.countplot(df[df["OnKameraMP"].isnull()]["CiftHat"]);
sns.countplot(df[df["OnKameraMP"].isnull()]["4G"]);
sns.distplot(df[df["OnKameraMP"].isnull()]["Agirlik"]);
df[df["OnKameraMP"].isnull()]["Agirlik"].mean()
df.groupby("Bluetooth").mean()
df.groupby("Bluetooth")[["OnKameraMP"]].mean()
df[(df["Bluetooth"] == "Yok") & (df["OnKameraMP"].isnull())]
bluetootholmayan_tabletler = df[(df["Bluetooth"] == "Yok") & (df["OnKameraMP"].isnull())].index

bluetootholmayan_tabletler
df.loc[bluetootholmayan_tabletler ,"OnKameraMP"] = 4 # 4 ile doldur,çünkü bluetooth'u olmayan telefonların OnKameraMP ortalaması 4.30
bluetootholan_tabletler = df[(df["Bluetooth"] == "Var") & (df["OnKameraMP"].isnull())].index
df.loc[bluetootholan_tabletler ,"OnKameraMP"] = 4 #4 değerini atarım çünkü Bluetooth özelliğine sahip tabletlerde ortalama 4.32'idi.
df.isna().sum()["OnKameraMP"]
df["OnKameraMP"] = df["OnKameraMP"].astype(int)
df[df["RAM"].isnull()]
df[df["RAM"].isnull()]["FiyatAraligi"]#Görüldüğü üzere hepsi pahalı tabletlerden oluşuyor.Bu da diğer tabletlere göre bu tabletlerin RAM özelliklerinin daha iyi olması gerektiğini düşünmeme sebep oluyor.
sns.countplot(df[(df["FiyatAraligi"] == "Pahalı")]  ["RAM"]);
df.groupby("FiyatAraligi")[["RAM"]].mean()#Pahalı olan tabletlerin RAM ortalamaları 3449.35'dir.Bu durumda bende eksik olan RAM değerlerimi bu değeri girebilirim.
EksikRAM_tabletler = df[(df["FiyatAraligi"] == "Pahalı") & (df["RAM"].isnull())].index

EksikRAM_tabletler
df.loc[EksikRAM_tabletler ,"RAM"] = 3450 # 3450 ile doldur, çünkü pahalı olan tabletlerin RAM ortalaması 3450 RAM küsürlü olamayacağı için int tipinde yuvarlayarak yazdım.
df.isna().sum()["RAM"] #Görüldüğü üzere eksik olan RAM değerlerini doldurmuş bulunuyoruz.
df["RAM"] = df["RAM"].astype(int)
df["RAM"].unique()
df["OnKameraMP"].unique()
from sklearn import preprocessing #İlk olarak label processing yapmak için kütüphanemi ekliyorum.
df["Bluetooth"].unique()
df["CiftHat"].unique()
df["4G"].unique()
df["3G"].unique()
df["Dokunmatik"].unique()
df["WiFi"].unique()
df["FiyatAraligi"].unique()
df["Renk"].unique()
%matplotlib inline

import matplotlib.pyplot as pl
#Farklı Değişkenler için farklı isimlerde sözlükler tanımlıyorum. 

varyok_mapping = {"Yok": 0, "Var": 1}   

FiyatAraligi_mapping = {"Çok Ucuz": 1, "Ucuz": 2, "Normal": 3, "Pahalı": 4}

df.head()

# Label encoding'de bazı değişkenler var yok cinsinde yazıldığı için Var:0 Yok:1 olarak alınıyordu.Bu bizim için sorun teşkil ediyor.Çünkü 1.değerin 0. değerden büyük olması gerekirken tam tersi bir durum oluşuyor.Bundan dolayı sözlük oluşturarak kendi belirlediğimiz şekilde sayısallaştırma tekniğini kullandım.

# FiyatAraligi değişkenimde de aynı sorun görülmekteydi.Bundan dolayı onun içinde ayrıca sözlük oluşturarak sayısallaştırma yöntemini kullandım.
#Evet-Hayır Sorularının Etiketlenmesi için 6 Değişkene bakıyor.

# bu kısımda ise tek tek onlara özel mapping'lerde bakılıp etiketleniyor.

def mymap(x, mapping): return mapping[x]

df['Bluetooth'] = df['Bluetooth'].apply(mymap, mapping = varyok_mapping)

df['CiftHat'] = df['CiftHat'].apply(mymap, mapping = varyok_mapping)

df['4G'] = df['4G'].apply(mymap, mapping = varyok_mapping)

df['3G'] = df['3G'].apply(mymap, mapping = varyok_mapping)

df['Dokunmatik'] = df['Dokunmatik'].apply(mymap, mapping = varyok_mapping)

df['WiFi'] = df['WiFi'].apply(mymap, mapping = varyok_mapping)

df['FiyatAraligi'] = df['FiyatAraligi'].apply(mymap, mapping = FiyatAraligi_mapping)
df.head() #Görüldüğü üzere Yok:0 Var:1 tüm değikenlerde yapıldı.Ayrıca FiyatAraligi ise ordinal olarak "Çok Ucuz": 1, "Ucuz": 2, "Normal": 3, "Pahalı": 4 olamk üzere sıralandı. 
df['Renk'] = pd.Categorical(df['Renk'])

dfDummies = pd.get_dummies(df['Renk'], prefix = 'Renk')

dfDummies
df = pd.concat([df, dfDummies], axis=1)

df.head()
df.drop(["Renk", "Renk_Yeşil"], axis = 1, inplace = True)

df.head()
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report

from sklearn.metrics import roc_auc_score,roc_curve

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import recall_score, f1_score, precision_score

from sklearn.tree import DecisionTreeClassifier

from warnings import filterwarnings

import matplotlib.pyplot as plt

from sklearn import ensemble

from sklearn.metrics import confusion_matrix as cm

from sklearn.metrics import mean_absolute_error #MAE için kütüphanemi ekliyorum.

from sklearn.metrics import mean_squared_error #MSE için kütüphanemi ekliyorum.

import math #RMSE için kütüphanemi ekliyorum.
x=df.drop("FiyatAraligi",axis=1)#x bağımsız değişkeni ifade ederken y bağımlı değişkeni ifade etmektedir.Bize verilen bilgilerde FiyatAraligi bağımlı değişken olduğu için y'ye atarken geri kalan özniteliklerimi x'e atadım.

y = df["FiyatAraligi"]
x #Bağımsız değişkenlerimi yazdırıyorum.
y #Bağımlı değişkenimi yazdırıyorum.
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score #Kütüphanemi dahil ediyorum.

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=42) #x,y train ve x,y test'lerden 4 parça oluşturuyorum.
x_test #x test değişkenimi kontrol ediyorum.
x_train #x train değişkenimi kontrol ediyorum.
y_test #y test değişkenimi kontrol ediyorum.
y_train #y train değişkenimi kontrol ediyorum.
from sklearn.naive_bayes import GaussianNB
# GaussianNB sınıfından bir nesne ürettik

gnb = GaussianNB()
# Makineyi eğitiyoruz

gnb.fit(x_train, y_train.ravel())
# Test veri kümemizi verdik ve iris türü tahmin etmesini sağladık

result = gnb.predict(x_test)
# Karmaşıklık matrisini yazdırıyorum.

cm = confusion_matrix(y_test,result)

print(cm)
# Başarı oranına bakıyoruz.

accuracy = accuracy_score(y_test, result)

print(accuracy)
y_pred=gnb.predict(x_test)
print(classification_report(y_test, y_pred))
F1Score = f1_score(y_test, y_pred, average = 'weighted')

F1Score
PrecisionScore = precision_score(y_test, y_pred, average='weighted')

PrecisionScore
RecallScore = recall_score(y_test, y_pred, average='weighted')

RecallScore
cart = DecisionTreeClassifier(random_state = 42, criterion='entropy')

cart_model = cart.fit(x_train, y_train)
cart_model

#Information Gain hesaplama için criterion='entropy'
!pip install skompiler

!pip install graphviz

!pip install pydotplus



from skompiler import skompile

print(skompile(cart_model.predict).to("python/code"))
df.columns
from sklearn.tree.export import export_text

r = export_text(cart, feature_names = ['BataryaGucu', 'Bluetooth', 'MikroislemciHizi', 'CiftHat', 'OnKameraMP',

       '4G', 'DahiliBellek', 'Kalinlik', 'Agirlik', 'CekirdekSayisi',

       'ArkaKameraMP', 'CozunurlukYükseklik', 'CozunurlukGenislik', 'RAM',

       'BataryaOmru', '3G', 'Dokunmatik', 'WiFi', 'Renk_Beyaz',

       'Renk_Gri', 'Renk_Kahverengi', 'Renk_Kırmızı', 'Renk_Mavi', 'Renk_Mor',

       'Renk_Pembe', 'Renk_Sarı', 'Renk_Siyah', 'Renk_Turkuaz',

       'Renk_Turuncu'])

print(r)
y_pred = cart_model.predict(x_test)
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
print(classification_report(y_test, y_pred))
F1Score = f1_score(y_test, y_pred, average = 'weighted')

F1Score
PrecisionScore = precision_score(y_test, y_pred, average='weighted')

PrecisionScore
RecallScore = recall_score(y_test, y_pred, average='weighted')

RecallScore
from sklearn.tree import export_graphviz

from sklearn import tree

from IPython.display import SVG

from graphviz import Source

from IPython.display import display

graph = Source(tree.export_graphviz(cart, out_file = None, feature_names = x.columns, filled = True))

display(SVG(graph.pipe(format = 'svg')))
ranking = cart.feature_importances_

features = np.argsort(ranking)[::-1][:10]

columns = x.columns



plt.figure(figsize = (16, 9))

plt.title("Karar Ağacına Göre Özniteliklerin Önem Derecesi", y = 1.03, size = 18)

plt.bar(range(len(features)), ranking[features], color="lime", align="center")

plt.xticks(range(len(features)), columns[features], rotation=80)

plt.show()
knn = KNeighborsClassifier(20)

knn_model = knn.fit(x_train, y_train)
knn_model
y_pred = knn_model.predict(x_test)
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
print(classification_report(y_test, y_pred))
F1Score = f1_score(y_test, y_pred, average = 'weighted')

F1Score
PrecisionScore = precision_score(y_test, y_pred, average='weighted')

PrecisionScore
RecallScore = recall_score(y_test, y_pred, average='weighted')

RecallScore
y_pred_KNN_Score = []

for k in range(2,15):

    module_KNN = KNeighborsClassifier(n_neighbors = k)

    module_KNN = module_KNN.fit(x_train, y_train)

    y_pred_KNN = module_KNN.predict(x_test)

    y_pred_KNN_Score.append(accuracy_score(y_test, y_pred_KNN))
for k in range(2,15):

    print( "Komşu sayısı" , k , "iken modelin başarı skoru :" , y_pred_KNN_Score[k- 2])
plt.figure(figsize=(15, 6))

plt.plot(range(2, 15), y_pred_KNN_Score, color='gold', linestyle='-', marker=".",

         markerfacecolor='orange', markersize = 12)

plt.title('Komşu Sayısına Göre KNN Modelinin Başarı Skoru ')

plt.xlabel('Komşu Sayısı')

plt.ylabel('Başarı Skoru')
import statsmodels.api as stat

stmodel = stat.OLS(y_train, x_train).fit()

stmodel.summary() #Burada tablo verimiz hakkında birçok bilgi içermektedir.Bağımlı değişkenimiz FiyatAraligi,R-squared değeri 0.986 gibi bilgileri elde ediyoruz.Güvenilirlik düzeyi 0.05'tir.

#Anlamlı olup olmadıklarını P>|t| değerlerine bakarak anlıyoruz.Eğer bu değer güvenilirlikten düşükse bu veriler anlamlıdır.Bizim modelimizde CozunurlukYükseklik,CozunurlukGenislik,RAM,BataryaGucu,Agirlik gibi değişkenlerimin değerleri 0.000 olduğu için bizim için çok anlamlı veirlerdir.Bunun yanı sıra P>|t|'si güvenilirlikten düşük olan diğer verilerimde çok anlamlı olmasada yinede anlamlıdır.