#Kutuphaneler

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from tensorflow.python.keras.utils import np_utils

#Veri Setimizi Projeye Dahil Edelim
dataset = pd.read_csv("../input/iris/Iris.csv")
#Veri setinin ilk 5 verisi yazdiralim.
dataset.head()
dataset = dataset.drop('Id', 1) ##Id sütununu eğitim sırasında bizim bir işimize yaramayacak. O yüzden onu silebiliriz.
dataset.head() #Id sütununu silebildik mi kontrol edelim.
#Her türden kaç tane veri olduğuna bakalım.
dataset["Species"].value_counts()
#Birde Grafik olarak bakalım.
sns.countplot(x='Species',data=dataset)
#Alt Yaprak Uzunluğuna göre dağılımın grafiğini inceleyelim.
sns.FacetGrid(dataset, hue="Species", size=5) \
   .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
   .add_legend()

plt.text(x=4.2, y=4.7, s='Alt Yaprak Uzunluğu ve Genisliği', fontsize=16, weight='bold')
plt.show()
#Taç Yaprak Uzunluğuna göre dağılımın grafiğini inceleyelim.

ax = sns.FacetGrid(dataset, hue="Species", size=5) \
   .map(plt.scatter, "PetalLengthCm", "PetalWidthCm") \
   .add_legend()

plt.text(x=1, y=2.8, s='Taç Yaprak Uzunluğu ve Genisliği', fontsize=16, weight='bold')
plt.show()

#iloc int bir değere erişmemizi sağlar. İlk 4 verimiz bağımsız değişken, 5. verimiz ise hedef değişkenimiz, yani bağımlı değişken. Bunları ayıralım.(Dizilerin 0'dan başladığını unutmayalım...)
X = dataset.iloc[:,0:4].values
#Hedef nitelik 4.indiste
y = dataset.iloc[:,4].values

#print(X)
#print(y)

#Deneyerek ayırma işlemini yapıp yapmadığımızı kontrol edebiliriz.
#Hedef degiskenimiz kategorik halde onuda sayısal hale çevirmeliyiz. Cünkü Ysa'ları sadece sayısal değerler ile çalışabilir. Bunun için  Label Encoder kullanacağız ve verimizi sayısal hale dönüştüreceğiz.
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
y = le.fit_transform(y)

#Kategorik degişkenden gölge degiskeni yaratalım.
y = np_utils.to_categorical(y)

#X'teki verilerimiz zaten sayısal olduğu için onlara LabelEncoder işlemi uygulamayacağız...

#Hedef değişkenlerimizi kategorik hale getirip getiremediğimizi kontrol edelim.

#print(y) ile bakabiliriz.

#İris Setosa = [1. 0. 0.]
#İris Versicolor = [0. 1. 0.]
#İris Virginica =  [0. 0. 1.]
#Veri setimizi eğitim ve test olarak ayıralım.(train_test_split kullanacağız)
from sklearn.model_selection import train_test_split
#Verilerin %20 sini test icin ayirdik. Geriye kalan %80'lik kısmıda eğitim verimiz olacak. RandomState'i verileri hangi indisten başlayarak ayırma işlemi yapacak gibi düşünebiliriz.Eğer bu şekilde kullanırsanız sizdeki eğitim ve test verileri ile benim eğitim ve test verilerim aynı olacaktır.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#Veri setimizi istediğimiz şekilde ayırabildik mi bir bakalım.
print("X_Train Veri Sayısı:",X_train.shape[0])
print("y_Train Veri Sayısı:",y_train.shape[0])
print("X_Test Veri Sayısı:",X_test.shape[0])
print("y_Test Veri Sayısı:",y_test.shape[0])

#Normalizasyon
#Başarı oranını arttırmak için verilerin normalize edilmiş hallerini kullanmamız gerekiyor. Bu işleme "Feature Scaling" yani Özellik Ölçeklendirme denir.
#Mesela MinMax Scaler kullanarak normalizasyon işlemi gerçekleştirilirken tüm verilerimizi 0 - 1 aralığına çekmiş olacağız. Veri Setimizideki en küçük sayısal değer 0'ı en büyük sayısal değer ise 1'i temsil edecek. Diğer değerler ise bu aralıkta yer alacaklar.

from sklearn.preprocessing import StandardScaler,MinMaxScaler

#MinMax veya Standart Scaler kullanabiliriz.
#scaler = MinMaxScaler() 
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#print(X_train) #Verilerimizin normalize edilmiş hallerine bir göz atalım.

#Gerekli kütüphaneleri ekleyeyim. 
import keras
from keras.models import Sequential 
from keras.layers import Dense

#Ysa'yi baslatalim
#Gizli Katman kaç tane eklemeliyiz ? 
#Kesin bir kural olmamakla birlikte genel olarak "(Girdi Sayısı + Çıktsı Sayısı) / 2" şeklinde kullanılıyor. Yani gizli katman sayımız : (4 + 3) / 2 = 4 (Yuvarlıyoruz)
model = Sequential()
model.add(Dense(4 ,input_dim=4, activation='relu')) #input_dim giriş katmanımızı ifade ediyor. 4 ise gizli katmandaki nöron sayısını.
model.add(Dense(3, activation = 'softmax')) #Çıktı Katmanı Ekleyelim. Hedef değişkenimiz 3 farkli değer alabildiği, çıktı katmanmız 3 nörondan oluşacaktır.Aktivasyon fonksiyonu olarak softmax kullanalım.

#Ysa'nin Derlenmesi
#optimizer: Stochastic Gradient Descenti gösteren "adam".
#loss: SGD’nin optimizasyonu için kullanılacak loss fonk. Tahmin y ile gerçek ye değeri arasını hesaplayıp en optimal değeri SGD’ye buldurur
#Hedef değişken sayımız 2'den fazla olduğu için categorical_crossentropy kullaniyoruz. Eğer 2 sınıf olsaydı binary_crossentropy kullanacaktık.

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
#Öğrenilmesi gereken parametre sayısı nasıl hesaplanıyor ona bakalım.
#Toplam Öğrenilmesi Gereken Parametre Sayısı = [GirişKatmanı x GizliKatman] + [GizliKatman x ÇıkışKatmanı] + Nöron Sayısı (Giriş Katmanı Hariç)
#Toplam Öğrenilmesi Gereken Parametre Sayısı = [4 x 4] + [ 4 x 3] + 4 + 3 = 35
model.summary() #Modelimizin Özetini Gösterir
#Modelimizi oluşturduk, şimdi eğitim zamanı !
#Epouch :Eğitim sırasında tüm eğitim verilerinin ağa kaç tur gösterileceğinin sayısıdır.
#Batch Size :Modelin aynı anda eğiteceği veri sayısıdır.
history = model.fit(X_train, y_train, batch_size=6, epochs=100)
#Modelimiz eğitim sürecini tamamladı. Şimdi ise daha önceden ayırdığımız %20'lik test verisi ile modelimizi test edelim.
scores = model.evaluate(X_test, y_test)
print("\nAccuracy( Doğruluk ): %",scores[1]*100) 
plt.figure()
plt.title('Model Başarısı')
plt.ylabel('Doğruluk')
plt.xlabel('Tur Sayısı')
plt.plot(model.history.history["accuracy"],label="Eğitim Doğruluk")
plt.plot(model.history.history["loss"],label="Eğitim Loss")
plt.legend()
plt.show()
#Başarıyı yukarıdaki şekilde hızlıca bulabiliriz ama birde Karmaşıklık Matrisi(Confusion Matrix) çizdirmek istersek bu şekilde yapalım.

y_pred = model.predict(X_test)
y_test_class = np.argmax(y_test,axis=1) #Daha önce hedef değişkenlerimizi [0,0,1], [0,1,0], [1,0,0] bu hale getirmiştik. Karmaşıklık Matrisinde kullanmak için 0,1,2 haline getirmemiz gerekiyor.
y_pred_class = np.argmax(y_pred,axis=1) 

print("Test Verileri:  ",y_test_class,"\nYSA Tahminleri: ",y_pred_class ) #Test verilerimiz ile tahminleri burada karşılaştırabiliriz.
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test_class, y_pred_class)) # Precision(Kesinlik) , Recall(Hassasiyet), F1-Score(F1-Değerlendirme)
cm = confusion_matrix(y_test_class, y_pred_class)
print(cm)
#Heatmap'le karmaşıklık matrisimizi(cm) görsel hale getirelim.
df_cm = pd.DataFrame(cm)
heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Greens") #Farklı renklerde görmek için cmap'i "BuPu","Blues","YlGnBu","Greens" yapabiliriz.