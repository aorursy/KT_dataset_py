import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
df = pd.read_csv("../input/tablet/tablet.csv").copy()
df.head() #Kamera değerlerinde 0 değerinin görülmesi o cihazın kamerasının olmadığını gösterir.
df.info()
df.shape
df.dtypes
df.describe().T
corr = df.corr() #0.8 ve üstü oranlar güçlü bir ilişki olduğunu gösterir fakat bu matris 0.8'in altında oranda kaldığından ilişkimiz güçlü değildir.

sns.heatmap(corr, xticklabels = corr.columns.values, yticklabels = corr.columns.values)

#CozunurlukGenislik ile CozunurlukYukseklik arasında orta derece pozitif korelasyon var.

#Renklendirmeler beyaza yaklaştıkça aralarındaki pozitif korelasyon katsayısı artıyor.

#onKameraMP ile ArkaKameraMP arasında orta derece pozitif bir korelasyon söz 

sns.countplot(x = "FiyatAraligi", data = df);
sns.set(rc={'figure.figsize':(10,7)}) #Oluşacak grafiklerin genişliği ve uzunluğu belirlendi.

sns.scatterplot(df["ArkaKameraMP"], df["FiyatAraligi"]);

#Arka kameranın çözünürlüğü bize fiyatla ilgili pek bir bilgi vermez. Çünkü seçilen herhangi bir kamera pahalı veya ucuz olabilir.
sns.barplot(x = "FiyatAraligi", y = "ArkaKameraMP", data = df);

#Bu grafikte üst üste binen nokta olup olmadığını inceledik. 
sns.scatterplot(x = "RAM", y = "FiyatAraligi", data = df);

#Bu grafik bize bellek miktarının tablet fiyatını etkileyebileceği konusunda bilgi verebilir.

#Grafikten çıkarabileceğimiz kadarıyla pahalı bir tabletin belleği minimum 2200 MB civarında iken bellek boyutu düştükçe fiyat aralığı normal, ucuz ve çok ucuz şeklinde değişiyor.
sns.scatterplot(x = "MikroislemciHizi", y = "FiyatAraligi", data = df);

#Mikroişlemci hızı değişmesine karşın her fiyat kategorisinde aynı işlemci hızından tabletler var.
sns.violinplot(x = "FiyatAraligi", y = "RAM", data = df);

#Bu grafik tablet belleğinin artması tabletin fiyatını arttırır tezini doğruluyor.
sns.violinplot(x = "FiyatAraligi", y = "MikroislemciHizi", data = df);

#Tabletlerin farklı işlemci hızlarına rağmen fiyat aralıklarındaki yoğunluk da hemen hemen aynı olduğundan mikroişlemci hızı bize fiyat-performans ürünü belirlemekte pek yardımcı olmaz.
sns.catplot("BataryaOmru", "FiyatAraligi", data = df, kind = "bar");

#Batarya Ömrü Bilgisi
sns.catplot("BataryaGucu", "FiyatAraligi", data = df, kind = "bar");

#Batarya gücü 1250 miliamperden yüksek ise ilgili tablet pahalı kategorisine giriyor ama bu değerden daha az miliamper içeren tabletler için kesin bir sonuç yok.
sns.scatterplot(x = "OnKameraMP", y = "FiyatAraligi", data = df);
sns.barplot(x = "FiyatAraligi", y = "OnKameraMP", data = df);
sns.countplot(df["FiyatAraligi"]); #Tam denge var.

#Makine öğrenmesinde amacımız tahmini fiyat aralığı olduğundan fiyat aralığında denge arıyoruz.
df["FiyatAraligi"].value_counts() 

# 4 fiyat kategorisi de 500'er adet tablet barındırıyor.
df.isnull().sum().sum()

#Toplamda 17 adet eksik gözlem var.
df.isnull().sum()

# 12 adet bellek bilgisi, 5 adet onkameraMP bilgisi eksik olan tablet var.
import missingno as msno #Eksik verileri daha iyi elde edebilmek için missingno modülünü çekirdeğe ekliyoruz.

msno.matrix(df,figsize=(20, 10));

#Eksik gözlemler arasında bir ilişki olup olmadığını anlamak için missingno matrisi çizdik.

#Aynı satırda oluşan beyaz çizgiler görseydik o zaman veriyi normalize etme amacıyla o tabletleri silme işlemi uygulayabilirdik.
msno.heatmap(df, figsize= (8,8));

#Heatmap metotu büyük veriler için faydalı olabilecek bir yöntem. Burada eksik gözlemler arasında veri güçlü bir ilişki yok.
def eksikDegerTablosu(df): 

    eksikDeger = df.isnull().sum()

    eksikDegerYuzde = 100 * df.isnull().sum()/len(df)

    eksikDegerTablo = pd.concat([eksikDeger, eksikDegerYuzde], axis=1)

    eksikDegerTablo_son = eksikDegerTablo.rename(columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})

    return eksikDegerTablo_son
eksikDegerTablosu(df)

#Yüzdeye baktığımızda bayağı az bir oranda eksik değer var. Eğer yüzde 30 ve civarında bir orandan daha fazla eksik gözlem olsaydı o tabletlerin eksik gözlemlerini doldurmak yerine silebilirdik.
df["OnKameraMP"].unique()
df.groupby("FiyatAraligi")[["OnKameraMP"]].mean()
cokUcuzOnKameralar = df[(df["FiyatAraligi"] == "Çok Ucuz") & (df["OnKameraMP"].isnull())].index

cokUcuzOnKameralar
df.loc[cokUcuzOnKameralar ,"OnKameraMP"] = 4.1 
ucuzOnKameralar = df[(df["FiyatAraligi"] == "Çok Ucuz") & (df["OnKameraMP"].isnull())].index

normalOnKameralar = df[(df["FiyatAraligi"] == "Normal") & (df["OnKameraMP"].isnull())].index

pahaliOnKameralar = df[(df["FiyatAraligi"] == "Pahalı") & (df["OnKameraMP"].isnull())].index
df.loc[ucuzOnKameralar ,"OnKameraMP"] = 4.3 

df.loc[normalOnKameralar ,"OnKameraMP"] = 4.5 

df.loc[pahaliOnKameralar ,"OnKameraMP"] = 4.3 
df.isna().sum()["OnKameraMP"]
df["OnKameraMP"].unique()
df["RAM"].unique()
df[df["RAM"].isnull()]

#Eksik değerlerin olduğu gözlemlere bakıyoruz fakat görünürde ortak özellikleri yok.
df.groupby("FiyatAraligi")[["RAM"]].mean()
cokUcuzBellekler = df[(df["FiyatAraligi"] == "Çok Ucuz") & (df["RAM"].isnull())].index

ucuzBellekler = df[(df["FiyatAraligi"] == "Ucuz") & (df["RAM"].isnull())].index

normalBellekler = df[(df["FiyatAraligi"] == "Normal") & (df["RAM"].isnull())].index

pahaliBellekler = df[(df["FiyatAraligi"] == "Pahalı") & (df["RAM"].isnull())].index
df.loc[cokUcuzBellekler ,"RAM"] = 785

df.loc[ucuzBellekler ,"RAM"] = 1679

df.loc[normalBellekler ,"RAM"] = 2582

df.loc[pahaliBellekler ,"RAM"] = 3449
df.isna().sum()["RAM"]
df["OnKameraMP"].unique()

#Eksik gözlem her zaman isna() fonksiyonu ile görüntülenemeyebilir. 

#Bazen eksik gözlem yerine boşluk karakteri girilebilir. 

#Bazen de eksik gözlem için "nan" gibi metinler girilebilir. 

#Bazen eksik gözlem yerine "boş", "yok" gibi metinler girilebilir. 

#O yüzden şüpheli gördüğümüz değişkenlerin benzersiz değerlerine bakarak sağlamasının yapılması gerekir.
from sklearn import preprocessing   # Veri ön işleme aşamasında label encoding vb. işlemler için çekirdeğe dahil ettik.

label_encoder = preprocessing.LabelEncoder()
df['3G'] = label_encoder.fit_transform(df['3G'])

df['4G'] = label_encoder.fit_transform(df['4G'])

df['Bluetooth'] = label_encoder.fit_transform(df['Bluetooth'])

df['CiftHat'] = label_encoder.fit_transform(df['CiftHat'])

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
X = df.drop(["FiyatAraligi"], axis=1) #Bağımsız değişkenler

y = df["FiyatAraligi"] #Bağımlı değişken 
y
X #Doğru şekilde ayrılmış 
from sklearn.model_selection import train_test_split 

#Verileri eğitime ve teste tabi tutmak amacı ile eklediğimiz modül



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

#Verilerimizin çeyreği test edilecek şekilde oranlandı. Kalan kısmı ile makine öğrenmesi yapılacak.
X_train
X_test
y_train
y_test # 4 parça değişkenimiz de istediğimiz şekilde ayrılmış.
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb_model = nb.fit(X_train, y_train)
nb_model
y_pred = nb_model.predict(X_test)
from sklearn.metrics import accuracy_score #Gerekli modülü ekledik



accuracy_score(y_test, y_pred)

#yüzde 75 doğruluk oranına sahip, yüksek değil belki fakat tatmin edebilecek bir oran.
from sklearn.metrics import confusion_matrix #gerekli modülümüzü ekledik

karmasiklikMatrisi = confusion_matrix(y_test, y_pred)

print(karmasiklikMatrisi)





#Her kolon bağımlı değişkenimiz olan fiyat aralığı için ucuzdan pahalıya için yapılan tahminler ile sıralanmış halde.

#Örneğin en alt satırı incelersek 118 doğru 11 adet yanlış tahmin yapılmış.

#En fazla yanlış tahmin en üst satır için yapılmış.

#Köşegen dışında kalan 0 dışındaki değerler yapılan yanlış tahminleri ifade ediyor.
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier

cart = DecisionTreeClassifier(random_state = 42)

cart_model = cart.fit(X_train, y_train)

#Default olarak (criterion = “gini”) parametre değerini kullandık. Bakalım skor değeri kaç olacak?
y_pred = cart_model.predict(X_test)
karmasiklikMatrisi = confusion_matrix(y_test, y_pred)

print(karmasiklikMatrisi)

#89 yanlış tahmin var.

#Bu tabloyu criterion='entropy' yapıp kıyas yapalım.
accuracy_score(y_test, y_pred)

# 0.822 oldu. 

# Şimdi bu değeri criterion='entropy' yapıp tekrar makine öğrenmesi uygulayıp o zaman oluşacak doğruluk skoru ile kıyaslayalım.
cart = DecisionTreeClassifier(random_state = 42, criterion='entropy')

#Default olarak (criterion = “gini”) parametresini kullanıyordu, entropy ile değiştirdik.

#Entropy belirsizlik arttıkça bilgi kazancının artmasını sağlıyor.

cart_model = cart.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
accuracy_score(y_test, y_pred)



#GaussianNB modeline kıyasla %14.4 daha fazla doğruluk skoru elde ettik.

#Default olarak kullanılan (criterion = “gini”) parametre değerine kıyasla doğruluk değerinde %4.37'lik artış oldu.
karmasiklikMatrisi = confusion_matrix(y_test, y_pred)

print(karmasiklikMatrisi)

#En fazla yanlış tahminin yapıldığı 2. satırda 19 yanlış var.

#Naive Bayes'de tek satırda yanlış tahmin sayısı 100'ü geçiyordu.

#Bu durumda çok daha başarılı bir model kullandığımızı söyleyebiliriz.

#Toplamda 71 yanlış tahmin var bu durum criterion = “gini” parametreli duruma kıyasla 18 adet daha fazla doğru tahminde bulunmuş.
print(classification_report(y_test, y_pred))

#Precision pozitif tahminde bulunduğumuz verilerin gerçekte hangi oranda pozitif olduklarını belirtir.

#En çok pozitif tahmin çok ucuz, en az pozitif tahmin normal fiyat diliminde.

#recall gerçekte pozitif olanların ne kadar doğru olduğunu ölçer.

#Gerçeğe çok yakın değerler olduğu görünüyor.

#f1 skoru precision ve recall değerlerinin ağırlıklı ortalamasıdır, daha kullanışlıdır.

#En iyi performans gösteren çok ucuz dilimidir.
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn_model = knn.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
accuracy_score(y_test, y_pred)

#0.93 değeri çok iyi denilebilecek bir değer.

#Doğruluk skoru bu modelde diğer modellere kıyasla en yüksek değere sahip oldu.
karmasiklikMatrisi = confusion_matrix(y_test, y_pred)

print(karmasiklikMatrisi)

#Toplamda 30 civarında yanlış tahmin var. Önceki modellere göre çok daha iyi sonuçlar elde ettik.
print(classification_report(y_test, y_pred))

#En iyi doğruluk performansı çok ucuz kategorisinde.
knn_params = {"n_neighbors": np.arange(2,16)}
from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, knn_params, cv = 3)

knn_cv.fit(X_train, y_train)
print("En iyi skor: " + str(knn_cv.best_score_))



#Skorumuz 0.934 fakat optimal komşu sayısı ile tekrardan modelleme yaparak yükseltebiliriz.



print("En iyi parametreler: " + str(knn_cv.best_params_))
karmasiklikMatrisi = confusion_matrix(y_test, y_pred)

print(karmasiklikMatrisi) #KNN ile kıyasla değişiklik elde edemedik, komşu sayısı belirleyip modelleme yapalım.
print(classification_report(y_test, y_pred)) #Burada da durum aynı.
score_list = []

for each in range(2,16,1):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(X_train,y_train)

    score_list.append(knn2.score(X_test, y_test))



plt.plot(range(2,16,1),score_list)

plt.xlabel("k en yakın komşu sayıları")

plt.ylabel("doğruluk skoru")



#Aldığımız en iyi skor 11 değerinde. O yüzden 11 değeriyle modelleme yapacağız.
knn = KNeighborsClassifier(11)

knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred)

#Optimal komşu sayısı ile yenden modellediğimizde doğruluk skorunda 0.04'lük bir artış sağladık.
karmasiklikMatrisi = confusion_matrix(y_test, y_pred)

print(karmasiklikMatrisi)

#Köşegen dışındaki sayı değerlerinin azaldığından dolayı görüldüğü gibi modelimiz artık daha doğru tahmin yapıyor.
print(classification_report(y_test, y_pred))

#En iyi doğruluk performansı normal kategorisinde.

#En düşük doğruluk performansı pahalı kategorisinde.