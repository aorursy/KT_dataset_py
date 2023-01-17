from IPython.display import Image # Image modulünü projeye dahil edilir.

Image("../input/knnimages/mantar.png") # Image fonksiyonuna resmin dosya yolu verilir.
import numpy as np # Kullanılacak olan dizi, matris veya vektör gibi lineer cebir araçları için gerekli olacaktır.

import pandas as pd # Veriyi düzenlemek, veriyi yüklemek ve bir dataframe yapısında veriyi daha kolay incelemek için gerekli olacaktır.
data=pd.read_csv("../input/mushroom-classification/mushrooms.csv")
mframe=pd.DataFrame(data)
mframe.head(10)
mframe.tail()
print(mframe.columns)
# Veri seti içerisinde yer alan "cap-shape","population","class" kolonlarının ilk 5 değeri getirilir.



print(mframe["cap-shape"].head()) 

print(mframe["population"].head())

print(mframe["class"].head()) 
Image("../input/knnimages/knn.png")
Image("../input/knnimages/uzaklikfonk.png")
X=mframe.iloc[:,1:] # "class" kolonu dışındaki tüm kolonlar girdi değeri olarak tanımlandı. Ne kadar çok birbiriyle alakalı girdi kolonu kullanılırsa çıktının tahmini o kadar yüksek doğrulukta çıkacaktır.



y=mframe.iloc[:,0]  # "class" kolonu çıktı değeri olarak tanımlandı. Çünkü class kolonu içerisindeki kategori tahmin edilecek.
from sklearn.preprocessing import LabelEncoder

from collections import defaultdict



d=defaultdict(LabelEncoder)



XFit=X.apply(lambda x: d[x.name].fit_transform(x))



LEncoder=LabelEncoder()



yFit=LEncoder.fit_transform(y)
import warnings # Uyarıların yönetimi için kullanılan kütüphanedir.



warnings.filterwarnings("ignore") # Çıkacak uyarıların gözardı edilmesi için kullanılır.



from sklearn.preprocessing import OneHotEncoder

ohc=defaultdict(OneHotEncoder)



resultFrame=pd.DataFrame()



kolonSayisi=mframe.shape[1]



for i in range(kolonSayisi-1):

    

    Xtemp_i=pd.DataFrame(ohc[XFit.columns[i]].fit_transform(XFit.iloc[:,i:i+1]).toarray())

    

    ohc_obj=ohc[XFit.columns[i]]

    LEncoder_i=d[XFit.columns[i]]

    Xtemp_i.columns=XFit.columns[i]+ "_" + LEncoder_i.inverse_transform(ohc_obj.active_features_)

    

    

    X_ohc_i=Xtemp_i.iloc[:,1:]

    

    resultFrame=pd.concat([resultFrame,X_ohc_i],axis=1)
print(mframe.shape,"->",resultFrame.shape) # Yeni veri setinin boyutu



# Bir kolonda yer alan kategoriler için ayrı ayrı kolonlar oluşturulmuş dolayısıyla veri setinin boyutu artmıştır.
resultFrame.head(10) # Hazırlanan yeni veri setinden örnek alınması.
from sklearn.model_selection import train_test_split # Eğitim ve Test verilerini ayırmak için kullanılan fonksiyondur.



X_train, X_test, y_train, y_test=train_test_split(resultFrame,yFit,test_size=0.3) # Test Verisi %30 Eğitim Verisi %70 olarak atandı.
from sklearn.neighbors import KNeighborsClassifier # KNN Algoritmasının modülü projeye dahil edildi.



KModel=KNeighborsClassifier(n_neighbors=30,metric="minkowski") # KNN Modeli kuruldu ve k komşuluk sayısı 30 olarak alındı. Yani sınıflandırılacak olan verinin 30 eleman komşuluğuna bakara karar verir. 

# Eğer herhangi bir uzaklık metriği verilmez ise algoritma "minkowski" uzaklığına göre uzaklık hesaplayacaktır. Burada da minkowski uzaklık metriği kullanılmıştır.



KModel.fit(X_train,y_train) # KNN algoritması ayarlanan Eğitim verileri üzerinde uygulandı ve model "Eğitildi"



y_pred=KModel.predict(X_test) # Uygulanan model için Test verileri tahmin edildi.
print(len(X_train))
Image("../input/knnimages/karisiklikmatrisi.png")
Image("../input/knnimages/dogrulukOrani.png")
from sklearn.metrics import confusion_matrix # Karışıklık matris fonksiyonunun dahil olduğu paket dahil edilir.



Karisiklik_Matrisi=confusion_matrix(y_test,y_pred) # Karışıklık matrisinin girdi değerleri yazılarak matris hesaplanır.



print(Karisiklik_Matrisi)
from sklearn.metrics import accuracy_score



dogruluk_Orani=accuracy_score(y_test,y_pred) # Yapılan sınıflandırma işleminin ne oranda doğru olduğunu döndürür. 0 ile 1 arasında değer alır.



print(dogruluk_Orani) 
Image("../input/knnimages/tprfpr.png")
from sklearn.metrics import roc_curve, roc_auc_score # ROC Eğrisi için gerekli değerlerin hesaplanmasında kullanılan modüllerdir.

import matplotlib.pyplot as plt # ROC Eğrisini çizmek için gerekli çizim modülüdür.





false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred) # roc_curve() fonksiyonu aldığı y_test ve y_pred değerlerini kullanarak FPR, TPR ve Threshold(Eşik Değeri) ifadelerini döndürür



print('KNN Algoritması için AUC Değeri : ', roc_auc_score(y_test, y_pred))
plt.subplots(figsize=(10,10))    

plt.title('ROC Eğrisi - KNN')

plt.plot(false_positive_rate, true_positive_rate)

plt.plot([0, 1], ls="--")

plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")

plt.ylabel('True Positive Rate (TPR)')

plt.xlabel('False Positive Rate (FPR)')

plt.show()
# Değişen k-komşuluk sayılarına göre algoritmanın doğruluk oranına bakılır:



def KNNHesapla(komsuluk,uzaklik,X_train,X_test,y_train,y_test):

    """

    Fonksiyon değişen komşuluk ve uzaklık parametrelerine göre KNN algoritmasının doğruluk oran listesini döndürür.

    

    komsuluk: KNN algoritmasında kullanılan k-komşuluk sayısının parametresidir.

    uzaklık: KNN algoritmasında kullanılan uzaklık tipinin parametresidir. minkowski - euclidean - manhattan olmak üzere üç türdür.

    X_train: Girdi değerlerinden eğitim verisi olarak kullanılan listedir.

    y_train: Çıktı değerlerinden eğitim verisi olarak kullanılan listedir.

    X_test: Girdi değerlerinden test verisi olarak kullanılan listedir.

    y_test: Çıktı değerlerinden test verisi olarak kullanılan listedir. Tahmin edilecek olan listedir.

    

    """

    oranListesi=[]

    

    for i in range(1,komsuluk):

        

        knnModel=KNeighborsClassifier(n_neighbors=i,metric=uzaklik) 

        

        knnModel.fit(X_train,y_train)

        

        y_pred=knnModel.predict(X_test)

        

        dogrulukOrani=accuracy_score(y_test,y_pred)

        

        oranListesi.append(dogrulukOrani)

        

    return oranListesi
# KNNHesapla fonksiyonu kullanılarak değişen uzaklık hesaplama yöntemlerine göre algoritmanın doğruluk oranları hesaplanır:



oranListe_Min=KNNHesapla(100,"minkowski",X_train,X_test,y_train,y_test) # Minkowski uzaklığı için 1-100 komşuluğunda algoritma doğruluk oranının listesi hesaplandı.



oranListe_Euc=KNNHesapla(100,"euclidean",X_train,X_test,y_train,y_test) # Euclidean uzaklığı için 1-100 komşuluğunda algoritma doğruluk oranının listesi hesaplandı.



oranListe_Man=KNNHesapla(100,"manhattan",X_train,X_test,y_train,y_test) # Manhattan uzaklığı için 1-100 komşuluğunda algoritma doğruluk oranının listesi hesaplandı.

plt.subplots(figsize=(15,15)) # Grafik çizimi için 15x15 boyutunda şablon oluşturulur.



x=range(1,100) # Grafikte kullanılacak olan X-Ekseni k-komşuluk sayısını temsil edeceğinden x-ekseni 1-100 arasında bir sayı dizisi olarak tanımlanır.



# X eksenleri sabit olacak şekilde daha öncesinden Minkowski, Euclidean ve Manhattan uzaklıkları kullanılarak hesaplanan doğruluk oranlarına göre farklı çizgiler aynı grafiğe eklenir:



plt.plot(x,oranListe_Min,color="r",linewidth=3.0,label="Minkowski",marker="o",markersize=5) 

plt.plot(x,oranListe_Euc,color="g",linewidth=3.0,label="Euclidean")

plt.plot(x,oranListe_Man,color="b",linewidth=3.0,label="Manhattan",linestyle="--")



# Grafiğin X ve Y ekseninde görünecek isimler ile bilgilendirme panosu eklenir: 



plt.xlabel("K-Komşuluk Değeri",fontsize="xx-large")

plt.ylabel("Algoritmanın Doğruluk Oranı",fontsize="xx-large")

plt.legend(fontsize="xx-large")



plt.show() # Grafikle birlikte bilgilendirme satırının çıkmamasını sağlar. Sadece grafiğin görseli çıktı olarak görünür.
