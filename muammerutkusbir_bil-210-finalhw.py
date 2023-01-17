import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df = pd.read_csv("../input/tablet/tablet.csv").copy()

df.head()
#kamera değerlerinde 0 yazılı olabiliyor . Bu durum aslında o cihazda kamera olmadığının göstergesi olabilir . 
#sıralama karışık bir şekilde olduğu görünüyor . 
df.info()
#8 object , 12 sayısal değerimiz var
#RAM ve OnKameraMP özniteliklerinde boş değerler var
#312 KB bellek kullanımı var
df.shape   # 20 öznitelik 2000 adet gözlem mevcut
df.dtypes
df["FiyatAraligi"].unique()
df.describe().T
#en sağdaki 5 kolonda kartillere bölünmüş halde değerleri gösteriyor
#%50 ile ifade edilen değerler medyan değerleri.
#standart sapma düştükçe bilgi kazancı azalır. Çünkü ayırt edicilik azalır. 
#Bu nedenle RAM , BataryaGücü , CozunurlukGenislik enformasyon açısından işimize yarayabilir  
#std sapma ortalamanın altına olması da iyiye işaret
corr = df.corr()
# 0.8 ve üstü oranlar güçlü bir ilişki olduğunu gösterir fakat incelediğimizde bu matriste güçlü bir ilişki söz konusu değil

sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)
#renklendirmeler beyaza yaklaştıkça aralarındaki pozitif korelasyon katsayısı artıyor
#arkaKameraMP ile önKameraMP arasında orta derece pozitif bir korelasyon söz konusu
#CozunurlukGenislik ile CozunurlukYukseklik arasında orta derece pozitif korelasyon var.
sns.countplot(x = "FiyatAraligi", data = df);
#gelen tabletlerimiz fiyatlandırma kategorisine göre 500 er tane eşit olarak dağılım göstermiş.
sns.countplot(df["Bluetooth"]);
#Bluetooth özelliği barındırmalarına göre tabletler neredeyse eşit sayıdalar
sns.countplot(df["WiFi"]);
#wifi özelliği barındırmalarına göre tabletler neredeyse eşit sayıdalar
sns.countplot(df["Renk"]);
#birbirlerine yakın fakat farklı renklerde farklı sayılarda tabletler var
sns.countplot(df["Dokunmatik"]);         
#neredeyse eşit sayıda  
sns.countplot(df["4G"]);     
#4G özelliği olan tabletler bir tık daha fazla
sns.countplot(df["CiftHat"]);    
#cift hatta sahip tabletler bir tık daha fazla           
sns.countplot(x="ArkaKameraMP",data=df)
#20 farklı megapixel değeri için farklı sayılarda tabletler var
sns.violinplot(df["MikroislemciHizi"]);
#0.5 mikroişlemci hızında çok fazla tablet bulunuyor
sns.countplot(x="OnKameraMP",data=df)
#iyi megapixel değerindeki tabler sayısı az
#düşük megapixel değerlerinde oldukça fazla tablet var
#en fazla ön kamerası olmayan tabletler mevcut
sns.violinplot(df["DahiliBellek"]);
#0-70 arasında hemen hemen aynı boyutta belleklere sahip tabletler var
sns.countplot(x="Kalinlik",data=df)
#en fazla 0.1 kalınlığında , en az 1 kalınlığında tabletler var
#Bu değerlerin arasında kalan kalınlık değerlerinde tablet sayıları değişkenlik gösteriyor
sns.violinplot(df["Agirlik"]);
#60-220 gram arasında tablet ağırlıkları değişiyor
#herhangi bir noktada yoğunluk yok
sns.countplot(x="CekirdekSayisi",data=df)
#birbirine benzer sayıda dağılmış
sns.violinplot(df["CozunurlukYükseklik"]);
#yaklaşık 300 değerlerinde yoğunluk mevcut
#sonrasında çözünürlükle beraber tablet miktarı azalmış
sns.countplot(x="ArkaKameraMP",data=df)
#5MP değerindeki tablet sayısı en az
#0-20 arasında arka kamera mp değerleri değişiyor
sns.countplot(x="BataryaOmru",data=df)
#2-20 arasında batarya ömürleri değişiyor 
#birbirine yakın sayıda değişkenlik göstermiş
sns.set(rc={'figure.figsize':(10,7)}) # oluşacak grafiklerin uzunluğunu ve genişliğini belirlendi.
sns.scatterplot(df["ArkaKameraMP"], df["FiyatAraligi"]);
#arka kameranın çözünürlüğü bize fiyatla alakalı pek bir bilgi vermiyor çünkü seçilen herhangi bir kamera ucuz da olabilir pahalı da
sns.violinplot(x = "FiyatAraligi", y = "ArkaKameraMP", data = df, height = 15, alpha = .5);
#genişlik olarak neredeyse aynı olduklarından pek bir şey ifade etmiyor kıyasladığımızda.
sns.barplot(x = "FiyatAraligi", y = "ArkaKameraMP", data = df);
#bu grafikte üst üste binen nokta olup olmadığını inceledik. 
sns.scatterplot(x = "RAM", y = "FiyatAraligi", data = df);
#bu grafikten anlayabileceğimiz pahalı bir tabletin rami minimum 2200 civarı mb boyutunda ve ram boyutu düştükçe fiyat aralığı normal , ucuz ve çok ucuz şeklinde gidiyor .
# bu grafik bize ram in tablet fiyatını etkileyebileceği konusunda bilgi verebilir.
sns.violinplot(x = "FiyatAraligi", y = "RAM", data = df, height = 15, alpha = .5);
#burada belli oluyor ki yüksek kapasiteli ramler daha yüksek fiyatlara satılıyor.
sns.barplot(x = "FiyatAraligi", y = "RAM", data = df);
#bu grafik ram in artmasının fiyatı artırması yönündeki etkisini kanıtlıyor.
sns.scatterplot(x = "MikroislemciHizi", y = "FiyatAraligi", data = df);
#mikroişlemci hızı değişmesine rağmen her fiyat kategorisinde aynı işlemci hızından tabletler var.
sns.violinplot(x = "FiyatAraligi", y = "MikroislemciHizi", data = df, height = 15, alpha = .5);
#fiyatlandırma açısından pek ayırt edilebilir bir görünüm yok
sns.barplot(x = "FiyatAraligi", y = "MikroislemciHizi", data = df);
#tabletlerde bulunan farklı işlemci hızlarına rağmen 
#fiyat aralığındaki yoğunluk da hemen hemen aynı olduğundan
# mikroişlemci hızı bize f/p ürünü belirlemekte pek katkı sağlamaz
sns.factorplot( "BataryaOmru", "FiyatAraligi", data = df, kind = "bar");
#batarya ömrü ortalamasının fiyat aralıklarına göre durumu 
sns.violinplot(x = "FiyatAraligi", y = "BataryaOmru", data = df, height = 15, alpha = .5);
#sadece çok ucuz olan görselde batarya ömrü yukarılara doğru azalmış
sns.factorplot( "BataryaGucu", "FiyatAraligi", data = df, kind = "bar");
#batarya gücü 1250 mah tan yüksek ise pahalı kategorisine giriyor 
#ama bu değerden az güçte batarya barındıran tabletler için kesin bir sonuç yok
sns.scatterplot(x = "OnKameraMP", y = "FiyatAraligi", data = df);
#hemen hemen her MP değeri  için her fiyat kategorisinde ürün mevcut
sns.violinplot(x = "FiyatAraligi", y = "OnKameraMP", data = df, height = 15, alpha = .5);
#ayırt edici görünüm yok
sns.barplot(x = "FiyatAraligi", y = "OnKameraMP", data = df);
#bize fiyatlandırma konusunda çok ayırt edici görünmüyor
sns.countplot(df["FiyatAraligi"]);#tam denge halinde
#makine öğrenmesinde amacımız fiyat aralığı tahmini olduğu içic fiyat aralığında denge arıyoruz.
df["FiyatAraligi"].value_counts() 
# 4 fiyat kategorisi de 500 er adet tablet barındırıyor.

df.isnull().sum().sum()
#toplamda 17 adet eksik değer bulunmakta
df.isnull().sum()
# 5 adet onkameraMP  bilgisi , 12 adet ram bilgisi eksik olan tablet var
import missingno # eksik verileri daha iyi okumak için missingno kütüphanesini ekliyoruz.
missingno.matrix(df,figsize=(20, 10));
#eksik değerler arasında bir ilişki olup olmadığını anlamak için missingno matrisi çizdik
#eğer aynı satırda oluşan beyaz çizgiler görseydik o zaman veriyi normalize etme amacıyla o tabletleri silme işlemi uygulayabilirdik
missingno.heatmap(df, figsize= (8,8));
#heatmap büyük veriler için faydalı olabilecek bir yöntem . burada eksik veriler arasında veri güçlü bir ilişki yok
def eksik_deger_tablosu(df): 
    eksik_deger = df.isnull().sum()
    eksik_deger_yuzde = 100 * df.isnull().sum()/len(df)
    eksik_deger_tablo = pd.concat([eksik_deger, eksik_deger_yuzde], axis=1)
    eksik_deger_tablo_son = eksik_deger_tablo.rename(columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return eksik_deger_tablo_son
eksik_deger_tablosu(df)
# yüzdeye baktığımızda çok çok az bir oranda eksik değer var
# eğer yüzde 30 civarı bi orandan daha fazla boş veri olsaydı o tabletlerin eksik değerlerini doldurmak yerine silebilirdik
df["OnKameraMP"].unique()
df.groupby("FiyatAraligi")[["OnKameraMP"]].mean()
CokUcuz_OnKameralar = df[(df["FiyatAraligi"] == "Çok Ucuz") & (df["OnKameraMP"].isnull())].index
CokUcuz_OnKameralar
df.loc[CokUcuz_OnKameralar ,"OnKameraMP"] = 4.1 
Ucuz_OnKameralar = df[(df["FiyatAraligi"] == "Çok Ucuz") & (df["OnKameraMP"].isnull())].index
Normal_OnKameralar = df[(df["FiyatAraligi"] == "Normal") & (df["OnKameraMP"].isnull())].index
Pahalı_OnKameralar = df[(df["FiyatAraligi"] == "Pahalı") & (df["OnKameraMP"].isnull())].index
df.loc[Ucuz_OnKameralar ,"OnKameraMP"] = 4.3 
df.loc[Normal_OnKameralar ,"OnKameraMP"] = 4.5 
df.loc[Pahalı_OnKameralar ,"OnKameraMP"] = 4.3 
df.isna().sum()["OnKameraMP"]
df["OnKameraMP"].unique()#ondalıklı sayılar var kamera değeri için çok fazla sıkıntı yaratmayacağından yuvarlamaya gerek yok
df["RAM"].unique()
df[df["RAM"].isnull()]
#eksik değerin olduğu gözlemlere bakıyoruz fakat görünürde ortak özellikleri yok
df.groupby("FiyatAraligi")[["RAM"]].mean()
CokUcuz_Ramler = df[(df["FiyatAraligi"] == "Çok Ucuz") & (df["RAM"].isnull())].index
Ucuz_Ramler = df[(df["FiyatAraligi"] == "Ucuz") & (df["RAM"].isnull())].index
Normal_Ramler = df[(df["FiyatAraligi"] == "Normal") & (df["RAM"].isnull())].index
Pahalı_Ramler = df[(df["FiyatAraligi"] == "Pahalı") & (df["RAM"].isnull())].index
df.loc[CokUcuz_Ramler ,"RAM"] = 785
df.loc[Ucuz_Ramler ,"RAM"] = 1679
df.loc[Normal_Ramler ,"RAM"] = 2582
df.loc[Pahalı_Ramler ,"RAM"] = 3449
df.isna().sum()["RAM"]
df["OnKameraMP"].unique()
#Eksik gözlem her zaman isna() fonksiyonu ile görüntülenemeyebilir. 
#Bazen eksik gözlem yerine boşluk karakteri girilebilir. 
#Bazen de eksik gözlem için "nan" gibi metinler girilebilir. 
#Bazen eksik gözlem yerine "boş", "yok" gibi metinler girilebilir. 
#O yüzden şüpheli gördüğümüz değişkenlerin benzersiz değerlerine bakarak sağlamasının yapılması gerekir.
from sklearn import preprocessing   # ön işleme aşamasında label encoding vb. için dahil ettik.
label_encoder = preprocessing.LabelEncoder()
df['Bluetooth'] = label_encoder.fit_transform(df['Bluetooth'])
df['CiftHat'] = label_encoder.fit_transform(df['CiftHat'])
df['4G'] = label_encoder.fit_transform(df['4G'])
df['3G'] = label_encoder.fit_transform(df['3G'])
df['Dokunmatik'] = label_encoder.fit_transform(df['Dokunmatik'])
df['WiFi'] = label_encoder.fit_transform(df['WiFi'])
df.head()
df['Renk'] = pd.Categorical(df['Renk'])
dfRenkKategorileri = pd.get_dummies(df['Renk'], prefix = 'Renk')
dfRenkKategorileri
df = pd.concat([df,dfRenkKategorileri],axis=1)
df.head()
df.drop(["Renk"], axis = 1, inplace = True)
df.head()
df.drop(["Renk_Yeşil"], axis = 1, inplace = True)
df.head()
y = df["FiyatAraligi"]#bağımlı değişkenimiz 

X = df.drop(["FiyatAraligi"], axis=1)#bağımsız değişkenlerimiz
y
X#doğru şekilde ayrılmış 
from sklearn.model_selection import train_test_split 
#verileri eğitime ve teste tabi tutmak amacı ile eklediğimiz kütüphane
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
#verilerimizin 1/4 ü test edilecek şekilde oranlandı. Kalan kısmı ile makine öğrenmesi yapılacak
X_train
X_test
y_train
y_test # 4 parça değişkenimiz de istediğimiz şekilde ayrılmış
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb_model = nb.fit(X_train, y_train)
nb_model
y_pred = nb_model.predict(X_test)
from sklearn.metrics import  accuracy_score#gerekli kütüphanemizi ekledik
accuracy_score(y_test, y_pred)
#yüzde 75 doğruluk oranına sahip ,  yüksek değil fakat tatmin edebilecek bir oran
from sklearn.metrics import confusion_matrix#gerekli kütüphanemizi ekledik
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)
print(karmasiklik_matrisi)
#köşegen dışında kalan 0 dışındaki değerler yapılan yanlış tahminleri ifade ediyor .
#her kolon bağımlı değişkenimiz olan fiyat aralığı için ucuzdan pahalıya için yapılan tahminler ile sıralanmış halde.
#örneğin en alt satırı incelersek 118 doğru 11 adet yanlış tahmin yapılmış
#en fazla yanlış tahmin en üst satır için yapılmış
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier(random_state = 42)
cart_model = cart.fit(X_train, y_train)
#default olarak (criterion = “gini”) parametre değerini kullandık bakalım skoru değeri kaç olacak
y_pred = cart_model.predict(X_test)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)
print(karmasiklik_matrisi)
#89 yanlış tahmin var
#Bu tabloyu criterion='entropy' yapıp kıyaslayalım 
accuracy_score(y_test, y_pred)
# 0.822 oldu. 
# Şimdi bu değeri criterion='entropy' yapıp tekrar makine öğrenmesi uygulayıp 
# o zaman oluşacak doğruluk skoru ile kıyaslayalım .
cart = DecisionTreeClassifier(random_state = 42, criterion='entropy')
#default olarak (criterion = “gini”) parametre değerini kullanıyordu biz entropy ile değiştirdik
#entopy belirsizlik arttıkça bilgi kazancının artmasını sağlıyor
cart_model = cart.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
accuracy_score(y_test, y_pred)
#GaussianNB modeline kıyasla % 14.4 daha fazla doğruluk skoru elde ettik .
#default olarak kullanılan (criterion = “gini”) parametre değerine kıyasla 
#doğruluk değerinde %4.37 lik artış oldu
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)
print(karmasiklik_matrisi)
#en fazla yanlış tahminin yapıldığı 2. satırda 19 yanlış var.b 
#Naive Bayes de tek satırda yanlış tahmin sayısı 100 ü geçiyordu.
#Bu durumda çok daha başarılı bir model kullandığımızı söyleyebiliriz.
#toplamda 71 yanlış tahmin var bu durum criterion = “gini” parametreli duruma kıyasla
#18 adet daha fazla doğru tahminde bulunmuş
print(classification_report(y_test, y_pred))
#precision pozitif tahminde bulunduğumuz verilerin gerçekte hangi oranda pozitif olduklarını belirtir.
#en çok pozitif tahmin çok ucuz , en az pozitif tahmin normal fiyat diliminde
#recall gerçekte pozitif olanların ne kadar doğru olduğunu ölçer
#gerçeğe çok yakın değerler olduğu görünüyor
#f1 skoru precision ve recall değerlerinin ağırlıklı ortalamasıdır daha kullanışlıdır .
#en iyi performans gösteren çok ucuz dilimi.
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
accuracy_score(y_test, y_pred)
#Doğruluk skoru bu modelde diğer modellere kıyasla en yüksek değere sahip oldu
# 0.93 değeri çok iyi denilebilecek bir değer.
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)
print(karmasiklik_matrisi)
#toplamda 30 civarında yanlış tahmin var . Önceki modellere göre çok daha iyi sonuçlar elde ettik.
print(classification_report(y_test, y_pred))
#en iyi doğruluk performansı çok ucuz kategorisinde
knn_params = {"n_neighbors": np.arange(2,16)}
from sklearn.model_selection import GridSearchCV
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, knn_params, cv = 3)
knn_cv.fit(X_train, y_train)
print("En iyi skor: " + str(knn_cv.best_score_))
#skorumuz 0.934 fakat optimal komşu sayısı ile tekrardan modelleme yaparak yükseltebiliriz
print("En iyi parametreler: " + str(knn_cv.best_params_))
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)
print(karmasiklik_matrisi)#KNN ile kıyasla değişiklik elde edemedik komşu sayısı belirleyip modelleme yapalım
print(classification_report(y_test, y_pred))#burada da durum aynı
score_list = []
for each in range(2,16,1):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(X_train,y_train)
    score_list.append(knn2.score(X_test, y_test))

plt.plot(range(2,16,1),score_list)
plt.xlabel("k en yakın komşu sayıları")
plt.ylabel("doğruluk skoru")
#aldığımız en iyi skor 11 değerinde . O yüzden 11 değeriyle modelleme yapacağız .

knn = KNeighborsClassifier(11)
knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
#optimal komşu sayısı ile yenden modellediğimizde doğruluk skorunda 0.04 lük bir artış sağladık 
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)
print(karmasiklik_matrisi)
#köşegen dışındaki sayı değerlerinin azaldığından görüldüğü gibi modelimiz artık daha doğru tahmin yapıyor.
print(classification_report(y_test, y_pred))
#en iyi doğruluk performansı normal kategorisinde
#en düşük doğruluk performansı pahalı kategorisinde