import numpy as np

import seaborn as sns

import pandas as pd

import os

import matplotlib.pyplot as plt

import statistics # medyan mod vs. için

from sklearn.tree import DecisionTreeClassifier # DecisionTree modeli için

from sklearn import ensemble

from sklearn.metrics import confusion_matrix as cm

from matplotlib.legend_handler import HandlerLine2D

from sklearn.neighbors import KNeighborsClassifier # 2-15 arası komşu sayılar için

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score # train test split için

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report#classification_report ve karmaşıklık matrisi

from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, precision_score # skor tahminleri için

from sklearn.naive_bayes import GaussianNB # Gaussian modelimiz için

from statistics import mode # mod alabilmek için

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
df = pd.read_csv("../input/tabletpc-priceclassification/tablet.csv") # Pandas kütüphanesi aracılığı ile datasetimizi tanıttık.

kopya = df.copy() # içinde değişikliğe gidersek kopyası elimizde her ihtimale karşı bulunsun...
df.sample(5) # Baktığımızda 12 adet sürekli verimiz var. Eksiksiz girilenlerde 2000'er tane değer tanımlı.
df.info() # Bu bilgilerden biz değişkenlerin tipini, bellek kullanımını, hatta kaçar tanesinin boş olmadığını görebiliyoruz.
df.shape # 20 özellik ve bu özelliklerden herhangi birine sahip 2000 tablet olduğunu öğrendik.
df.describe() # 12 sürekli değişkenimize dair bilgler görüntülenmektedir. Bilgilerde;

# count = kaç adet olduğu

# mean = ortalaması

# std = standart sapması

# min = en küçük değer, max = en büyük değer , ve çeyrekler açıklıkları olmak üzere listelenmiştir.

# ÖNEMLİ UYARI : import statistics dedikten sonra statistics.mean(df["RAM"]) yazarsak çıktı olarak nan verecektir. Çünkü

# RAM özelliğinde nan yani boş bırakılanlar var, bunu statistics olarak hesaplamaktansa burdan erişmek daha uygun.
print(pd.concat([df.isnull().sum(), 100 * df.isnull().sum()/len(df)], 

          axis = 1).rename(columns={0:'Kayıp Değerler', 1:'Yüzdesi (%)'}))



print('\nTOPLAM NULL DEGER SAYISI : \t', df.isna().sum().sum(), '\t   ',

     100 * df.isnull().sum().sum()/len(df))

# 12 RAM ve 5 Ön kameranın kaç MP olduğunu bilmiyoruz. 2000 tablet içinde 17 tanesinin boş olması güzel bir

# veri setine sahip olduğumuzu gösterir. Bu değerleri doldurmanın birçok yolu var fakat ortalama bir değeri bu 

# özelliklere atarsak ortalamada değişiklik olmayacağı ve bizden de doldurmamız istendiği için ben ilerleyen adımlarda

# ortalama değer atayacağım.
print(df["FiyatAraligi"].unique()) # Kaç çeşit kategoriye sahip olduğumuzu

print(df["FiyatAraligi"].nunique()) # ve toplamda kaç kategori olduğunu böyle öğrenebiliriz

print(df["Renk"].unique())

print(df["Renk"].nunique())
df.corr() # Tüm alanların birbiri ile korealasyonun getirir. 1'e çok yakın olmasa da en güçlü ilişki 0.64 ile ArkaKameraMP ve

# OnKameraMP arasındadır. Yani eğer tablet alıp güzel manzaralar çekmek isterseniz, güzel selfie'ler de çekebilirsiniz :))
corr = df.corr() # df.corr()'dan yararlanarak ısı haritamızı çiziyoruz. Yüzdeliklerini görüntülemek daha kolay yorum 

sns.heatmap(corr, # yapmamızı sağlar. Yukarıda aramak yerne en güçlü ilişkiyi burdan tespit etmek işimizi daha da kolaylaştırır.

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values, annot=True, fmt='.0%');
sns.countplot # yazarak içine yazabileceğimiz parametreleri görebiliriz.
df.hist(figsize = (15,15),color="c")



plt.show()  # Burada bakacak olursak en dengeli Çözünürlük Genişlik gibi gözüküyor, OnKameraMP ise en dengesiz verimiz gibi

# gözüküyor
print(df["FiyatAraligi"].value_counts())

print(sns.countplot(x="FiyatAraligi", data = df)) # 500x4 ile dengeli dağıldığını öğrenmiş ve görmüş olduk...
renkler = ["white", "pink", "purple", "orange", "grey", "yellow", "darkblue", "turquoise", "brown", "green", "red", "black"];

print(sns.countplot(x="Renk", data = df, palette = renkler));

print(df["Renk"].value_counts()) # Isimler sığmadığı için renklerden de anlaşılmasını istedim. List tanımlayıp içine renkleri

# sırasıyla yazıp bunu da palette deişkenine atadım.
sns.countplot(x="WiFi", data = df)

print(df["WiFi"].value_counts()) # Tüm veriler ile ilgili açıklamalar en sonda yapılmıştır...
print(df["Dokunmatik"].value_counts())

sns.countplot(x="Dokunmatik", data = df)
sns.countplot(x="3G", data = df)

print(df["3G"].value_counts())
sns.countplot(x="4G", data = df)

print(df["4G"].value_counts())
sns.countplot(x="CiftHat", data = df)

print(df["CiftHat"].value_counts())
sns.countplot(x="Bluetooth", data = df)

print(df["Bluetooth"].value_counts())
sns.violinplot(x = df.FiyatAraligi, y = df.RAM, data = df);
sns.catplot(x="FiyatAraligi", y="MikroislemciHizi", hue="4G",

            kind="swarm", data=df);
sns.relplot(x="FiyatAraligi", y="CekirdekSayisi", col="Dokunmatik",hue="DahiliBellek", data = df); # Bu sefer Dokunmatik olan 

# olmayan tabletler için Çekirdek Sayısı artarken Dahili Bellek sayısındaki artışı Fiyat Aralığına göre inceliyoruz. 

# 4 türü aynı anda yorumlayabiliyoruz fakat biraz daha karışık.
# NAN yani boş bırakılan değerlerinkini bularken nan çıktısını vereceği için, nan olan RAM ve 

# OnKameraMP özelliklerine yukarıda yazdığım describe()'tan bakmak daha uygun olacaktır.
print(statistics.median(df["ArkaKameraMP"]))

print(mode(df["Renk"])) # moduna bakabilmemiz için değerlerin dengeli dağılmamaları lazım, bu yüzden Fiyat aralığının moduna

# bakamayız

print(mode(df["4G"]))
kopya['RAM'] = kopya['RAM'].fillna(kopya['RAM'].sum()/len(kopya['RAM'])) # aslı nolur nolmaz bende kalsın dedim ve 

kopya['OnKameraMP'] = kopya['OnKameraMP'].fillna(kopya['OnKameraMP'].sum()/len(kopya['OnKameraMP'])) # ortalamalarını aldım

# ve boş değerlere onları atadım. Test ederek görebiliriz;
kopya.info() # görüldüğü gibi hiç boş değer bulunmamakta artık.
kopya.Bluetooth = kopya.Bluetooth.eq('Var').mul(1) # Var olanları 1'e eşitleyecek, diğerlerini 0'a ve sonrasında bunu

kopya.Bluetooth # kopya.Bluetooth'a atayacak. Böylelikle Bluetooth özelliğini sayısallaştırmış olduk. Diğerlerine de uygulayalım
kopya['CiftHat'] = kopya['CiftHat'].map({'Var': 1, 'Yok': 0}) # eq().mul()'dan farkı değişecek değerleri kendimiz girebiliyoruz.

kopya.CiftHat # Yani 2 seçenekten fazlası var ise elimizde(renkler gibi) bu yöntem kullanışlı olacaktır.
kopya['4G'] = kopya['4G'].map({'Var': 1, 'Yok': 0})

kopya['4G']
kopya['3G'] = kopya['3G'].map({'Var': 1, 'Yok': 0})

kopya['3G']
kopya['Dokunmatik'] = kopya['Dokunmatik'].map({'Var': 1, 'Yok': 0})

kopya['Dokunmatik']
kopya['WiFi'] = kopya['WiFi'].map({'Var': 1, 'Yok': 0})

kopya['WiFi']
kopya['FiyatAraligi'] = kopya['FiyatAraligi'].map({'Çok Ucuz': 0, 'Ucuz': 1, 'Normal': 2, 'Pahalı': 3})

kopya['FiyatAraligi'] # Fiyat aralıkları sıralanabilir(ordered) yani Ordinal'dir.
renkler = ["white", "pink", "purple", "orange", "grey", "yellow", "darkblue", "turquoise", "brown", "green", "red",

           "darkslategray"];

sns.violinplot(x="Renk", y="FiyatAraligi", data = kopya, height = 8, alpha = .5, palette = renkler);
Renkler = pd.get_dummies(kopya["Renk"]) # Renkleri sayısallaştırdık

Renkler.head() # artık yeni renklerimiz veri setimizde böyle görünecek
kopya = pd.concat([kopya, Renkler], axis = 1) # artık Renk kategorisi yerine dummy olarak elde ettiğimiz Renkler bulunmakta

kopya.drop(["Renk"], axis = 1, inplace = True) # Bu sayede 20 özniteliğe 12 yenisi eklendi ve 1 tanesi çıkarıldı

kopya.head() # bu sayede toplamda 31 sütunumuz olması gerekiyor.
kopya.hist(figsize = (15,15),color="brown")



plt.show() # Tüm sayısal verilerimiz aşağıda modellenmiştir.
Y = kopya.iloc[:,18].values # 18. sütunu almak için bu kodu yazıyoruz. Renkler en sona eklendiği için ilk 19 sütun aynı.

X = kopya.drop(['FiyatAraligi'], axis=1) # kopya verisetinden Y için kullandığımız sütunu atıp X için kullanıyoruz.
Y # 'Çok Ucuz': 0, 'Ucuz': 1, 'Normal': 2, 'Pahalı': 3 olmak üzere;
X # kopya.head(5) yazarak doğru olup olmadığını kontrol edelim.
kopya.head(5) # verilerimiz başarılı bir şekilde bağımlı ve bağımsız olarak ayrıldığını gördük.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42) # Şu an %75 eğitim ve %25 test

# olmak üzere değişkenlerimizi train_test_split metoduyla ayırdık. Verileri de bize rastgele verdi.
X_train  # gördüğümüz gibi 31-1=30 sütunumuzda 1500 örnek mevcut. Bu değerleri kendisini eğitmesi için makineye göndereceğiz.
X_test # 500x30 yani 500 tanesi test için kullanılacak. Bunları modele göndereceğiz ve ondan Y'yi tahmin etmesini isteyeceğiz.
Y_train
Y_test # Tahmin ettiği Y değerleri ile gerçek değerlerimizi karşılaştıracağız ve başarı oranına bakacağız.
nb = GaussianNB()

nb_model = nb.fit(X_train, Y_train) # Modelimize train için ayırdığımız verileri gönderiyoruz. Test kısmı için ayrılan %25 lik

# kısımdan haberi yok. Tahmin etmesini isteyeceğiz ve kalan %25 ile karşılaştıracağız.
X_test[0:10]
nb_model.predict(X_test)[0:10]
Y_test[0:10] # Gördüğümüz gibi üstte ve altta değerlerimizi karşılaştırabiliriz. Ben şöyle bir yorumda bulundum. Değerler 0 dan

# 4'e doğru arttıkça yani çok ucuzdan pahalıya doğru makine tahminde hatalar yapıyor.
Y_pred = nb_model.predict(X_test)

Y_pred
Y_test # karşılaştırmamızı daha kolay yapabiliriz.
print(accuracy_score(Y_test, Y_pred))

print(nb_model.score(X_test, Y_test))
karmasiklik_matrisi = confusion_matrix(Y_test, Y_pred)

print(karmasiklik_matrisi) 
(karmasiklik_matrisi[0][0] + karmasiklik_matrisi[1][1] + karmasiklik_matrisi[2][2] + karmasiklik_matrisi[3][3]) / ((karmasiklik_matrisi[0][0] + karmasiklik_matrisi[0][1] + karmasiklik_matrisi[0][2] + karmasiklik_matrisi[0][3]) + (karmasiklik_matrisi[1][0] + karmasiklik_matrisi[1][1] + karmasiklik_matrisi[1][2] + karmasiklik_matrisi[1][3]) + (karmasiklik_matrisi[2][0] + karmasiklik_matrisi[2][1] + karmasiklik_matrisi[2][2] + karmasiklik_matrisi[2][3]) + (karmasiklik_matrisi[3][0] + karmasiklik_matrisi[3][1] + karmasiklik_matrisi[3][2] + karmasiklik_matrisi[3][3]))

# Artılardan sonra aşağı geçip düzenli yazmak istediğinizde hata veriyor hata ile uğraşırken böyle deneyince biraz güldüm :)))

# Bu skoru şöyle elde ediyoruz : TP+TN / TP+TN+FN+FP . Şöyle de akılda tutabilirsiniz : Köşegenlerin toplamı / Tümü
print(classification_report(Y_test, Y_pred)) # Gördüğümüz gibi yukarıda yaptığım yorumu kanıtlar nitelikte oldu. Çok ucuzdan

# pahalıya gittikçe modelimiz tahminlerinde hatalar yaptı. Ortalama skorları da görebiliyoruz.

# Modelimiz 0.91 e çıkabilmiş fakat 0.60'ı da görmüş. Birbirine yakın ve yüksek değerler her zaman daha iyidir.
F1Score = f1_score(Y_test, Y_pred, average='weighted')  

F1Score
PrecisionScore = precision_score(Y_test, Y_pred, average='weighted')

PrecisionScore
RecallScore = recall_score(Y_test, Y_pred, average='weighted')

RecallScore
cart_grid = {"max_depth": range(1,20), "min_samples_split" : range(2,50)}

# Modelimizi inşa edelim ve eğitelim. Ancak modeli inşa etmeden önce model için kritik olan iki parametreyi optimize edelim.
cart = DecisionTreeClassifier() # modelimizi oluşturuyoruz

cart_cv = GridSearchCV(cart, cart_grid, cv = 10, n_jobs = -1, verbose = 2)
cart_cv_model = cart_cv.fit(X_train, Y_train) # modele değerlerimizi gönderdik. Bitirme sürelerini de ekranda görebiliriz.
print("En iyi parametreler : " + str(cart_cv_model.best_params_))

print("En iyi skor : " + str(cart_cv_model.best_score_))
cart = DecisionTreeClassifier(max_depth = 13, min_samples_split = 11) # Buraya en iyi parametrelerimizi yazarsak en iyi skoru 

cart_tuned = cart.fit(X_train, Y_train) # yakalamış oluruz. Bu işlemi manuel yapmalıyız.
Y_pred = cart_tuned.predict(X_test)
accuracy_score(Y_test, Y_pred) # Bir önceki modellemeden daha yüksek bir skor verdi
karmasiklik_matrisi = confusion_matrix(Y_test, Y_pred)

print(karmasiklik_matrisi) 
(karmasiklik_matrisi[0][0] + karmasiklik_matrisi[1][1] + karmasiklik_matrisi[2][2] + karmasiklik_matrisi[3][3]) / ((karmasiklik_matrisi[0][0] + karmasiklik_matrisi[0][1] + karmasiklik_matrisi[0][2] + karmasiklik_matrisi[0][3]) + (karmasiklik_matrisi[1][0] + karmasiklik_matrisi[1][1] + karmasiklik_matrisi[1][2] + karmasiklik_matrisi[1][3]) + (karmasiklik_matrisi[2][0] + karmasiklik_matrisi[2][1] + karmasiklik_matrisi[2][2] + karmasiklik_matrisi[2][3]) + (karmasiklik_matrisi[3][0] + karmasiklik_matrisi[3][1] + karmasiklik_matrisi[3][2] + karmasiklik_matrisi[3][3]))
cross_val_score(cart_tuned, X_test, Y_test, cv = 10)

cross_val_score(cart_tuned, X, Y, cv = 10).mean() # 10 defa yaptık ve 10 a bölerek ortalamasını bulduk
PrecisionScore = precision_score(Y_test, Y_pred, average='weighted')

PrecisionScore 
RecallScore = recall_score(Y_test, Y_pred, average='weighted')

RecallScore # En yüksek skoru burdan aldık
F1Score = f1_score(Y_test, Y_pred, average = 'weighted')  

F1Score
print(classification_report(Y_test, Y_pred)) # Tahminimiz Recall için 0.92'den 0.79 arasında oynamış. Bu skorlar çok iyi.
from sklearn import tree

DT = tree.DecisionTreeClassifier(criterion = 'entropy')

DT_model = DT.fit(X_train,Y_train)

DT_Y_pred = DT_model.predict(X_test)

kopya_DT = pd.DataFrame({ "Tahmini " : DT_Y_pred, "Gerçek " : Y_test, "Sonuc " : DT_Y_pred == Y_test})

kopya_DT # Buradan hangi değerleri doğru tahmin ettiğini görüyoruz. Burdan pek anlamasak da yeni bir skor isteyerek karşılaştırma yapabiliriz.
DT_model.score(X_test,Y_test) # Görüldüğü üzere skorumuz yükseldi. criterion = “entropy” olarak kullanmak bizim işimize yaradı.
print(classification_report(Y_test, DT_Y_pred)) # Aralığımız 0.93 ile 0.78'e yükselebildi. F1-score daha yakın skorlar verdi.
karmasiklik_matrisi = confusion_matrix(Y_test, DT_Y_pred)

print(karmasiklik_matrisi)

(karmasiklik_matrisi[0][0] + karmasiklik_matrisi[1][1] + karmasiklik_matrisi[2][2] + karmasiklik_matrisi[3][3]) / ((karmasiklik_matrisi[0][0] + karmasiklik_matrisi[0][1] + karmasiklik_matrisi[0][2] + karmasiklik_matrisi[0][3]) + (karmasiklik_matrisi[1][0] + karmasiklik_matrisi[1][1] + karmasiklik_matrisi[1][2] + karmasiklik_matrisi[1][3]) + (karmasiklik_matrisi[2][0] + karmasiklik_matrisi[2][1] + karmasiklik_matrisi[2][2] + karmasiklik_matrisi[2][3]) + (karmasiklik_matrisi[3][0] + karmasiklik_matrisi[3][1] + karmasiklik_matrisi[3][2] + karmasiklik_matrisi[3][3]))
knn = KNeighborsClassifier()
knn_tuned = knn.fit(X_train, Y_train)



Y_pred = knn_tuned.predict(X_test)
accuracy_score(Y_test, Y_pred) # Bu model içlerinde en yüksek skoru veren model oldu.
karmasiklik_matrisi = confusion_matrix(Y_test, Y_pred)

print(karmasiklik_matrisi)
(karmasiklik_matrisi[0][0] + karmasiklik_matrisi[1][1] + karmasiklik_matrisi[2][2] + karmasiklik_matrisi[3][3]) / ((karmasiklik_matrisi[0][0] + karmasiklik_matrisi[0][1] + karmasiklik_matrisi[0][2] + karmasiklik_matrisi[0][3]) + (karmasiklik_matrisi[1][0] + karmasiklik_matrisi[1][1] + karmasiklik_matrisi[1][2] + karmasiklik_matrisi[1][3]) + (karmasiklik_matrisi[2][0] + karmasiklik_matrisi[2][1] + karmasiklik_matrisi[2][2] + karmasiklik_matrisi[2][3]) + (karmasiklik_matrisi[3][0] + karmasiklik_matrisi[3][1] + karmasiklik_matrisi[3][2] + karmasiklik_matrisi[3][3]))
cross_val_score(cart_tuned, X_test, Y_test, cv = 10)

cross_val_score(cart_tuned, X, Y, cv = 10).mean() 
PrecisionScore = precision_score(Y_test, Y_pred, average='weighted')

PrecisionScore
RecallScore = recall_score(Y_test, Y_pred, average='weighted')

RecallScore
F1Score = f1_score(Y_test, Y_pred, average = 'weighted')  

F1Score
print(classification_report(Y_test, Y_pred)) # Skorumuz 0.98 e kadar çıkmış. Büyük başarı. En iyi modelleme içlerinde bu oldu...
knn_params = {"n_neighbors": np.arange(2,15)}





knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, knn_params, cv = 3)

knn_cv.fit(X_train, Y_train)
print("En iyi skor: " + str(knn_cv.best_score_))

print("En iyi parametreler: " + str(knn_cv.best_params_)) # en iyi komşu sayısı 0.928 puan ile 11'dir. Fakat bunu yazdırırken 9 olarak gösteriyor.

# Bunun sebebi 2 den başlıyor olması, o yüzden 2 sayı geri atmış.
knn = KNeighborsClassifier(11) # n_neighbors un bulduğu en iyi değeri içine yazdırıyoruz.

knn_tuned = knn.fit(X_train, Y_train)



Y_pred = knn_tuned.predict(X_test)



accuracy_score(Y_test, Y_pred) #  En iyi neighbor değeri ile 0.92'den 0.93'e yükseldi.
karmasiklik_matrisi = confusion_matrix(Y_test, Y_pred)

print(karmasiklik_matrisi)
(karmasiklik_matrisi[0][0] + karmasiklik_matrisi[1][1] + karmasiklik_matrisi[2][2] + karmasiklik_matrisi[3][3]) / ((karmasiklik_matrisi[0][0] + karmasiklik_matrisi[0][1] + karmasiklik_matrisi[0][2] + karmasiklik_matrisi[0][3]) + (karmasiklik_matrisi[1][0] + karmasiklik_matrisi[1][1] + karmasiklik_matrisi[1][2] + karmasiklik_matrisi[1][3]) + (karmasiklik_matrisi[2][0] + karmasiklik_matrisi[2][1] + karmasiklik_matrisi[2][2] + karmasiklik_matrisi[2][3]) + (karmasiklik_matrisi[3][0] + karmasiklik_matrisi[3][1] + karmasiklik_matrisi[3][2] + karmasiklik_matrisi[3][3]))
PrecisionScore = precision_score(Y_test, Y_pred, average='weighted')

PrecisionScore
RecallScore = recall_score(Y_test, Y_pred, average='weighted')

RecallScore
F1Score = f1_score(Y_test, Y_pred, average = 'weighted')  

F1Score
print(classification_report(Y_test, Y_pred)) # Skorlarımız çok yükseldi görüldüğü gibi.
score_list = []



for each in range(2,16):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(X_train,Y_train)

    score_list.append(knn2.score(X_test, Y_test))

    

plt.plot(range(2,16),score_list)

plt.xlabel("komşu sayıları")

plt.ylabel("doğruluk skoru")

plt.show() 
array = []

i = 2

while i < 16 :

    knn = KNeighborsClassifier(n_neighbors = i)

    knn_tuned = knn.fit(X_train, Y_train)

    Y_pred = knn_tuned.predict(X_test)

    print(i,"Komşulu Skorumuz : ", knn_tuned.score(X_test,Y_test))

    knn_score = knn_tuned.score(X_test, Y_test)

    array.insert(i,knn_score)

    i = i + 1  # 11 sayısı bu veri seti için en iyi komşu sayısı