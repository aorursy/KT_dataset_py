#İlgili kütüphanelerimizi projemize import  ediyoruz

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import os 
print(os.listdir("../input"))


#bu verisetinde  eski bir kızılderili kabilesinin verilen sağlık bilgileriyle diyabet hastası olup olmadığını test etmeye çalışacağız
data = pd.read_csv('../input/PimaIndians.csv')
#Burada datanın matematiksel değerleri gözükmektedir.
data.describe()
#Tüm değerlerin dolu olduğunu görüyoruz bu sayede verimizden daha iyi sonuçlar alacağız ve preprocessing bölümü biraz daha kolaylaşmış oldu
#Toplamda 9 tane feature değerimiz var 'test' bölümü ise bizim sonuç bölümümüz 

data.info()
#Pregnat: Kaç kere hamile kaldığı
#Glucose: Oral glikoz tolerans testinde 2 saat içerisindeki plazma glukoz konsantrasyonu
#diastolic(BloodPressure): Diyastolik kan basıncı (mm Hg)
#triceps(SkinThickness): Triceps cilt kat kalınlığı (mm)
#Insulin: 2 saatlik  insülin serumu (mu U/ml)
#BMI: Vücut kitle indeksi
#diabetes(DiabetesPedigreeFunction): Diyabet pedigree işlevi
#Age: Yaş (yıl)
#test(Outcome): Sınıf Değişkeni (0 veya 1)

#İlk 5 değerimizi inceliyoruz. 
#Özellikle insulin değerinde katlı oranlar görülmekte ancak 0. ve 2. indis incelendiğinde rakamlar birbirine çok yakın ama sonuçlar farklı gözlenmiştir.
data.head()

#Son beş değer izlendiğinde ise glucose, diastolic verileri daha mantıklı sonuçlara ulaşmamız için yardımcı olabilir duruyor.
data.tail()
#datamız 392 satır(değer) 9 sütundan(feature) oluşmaktadır. 
data.shape 
#Burada histogram grafiği ile teste dahil olan yaş gruplarını görmeye çalıştık.
#Sonuçlardan da anlaşılacağı gibi datamızın çoğunluğu genç diyebileceğimiz kişilerden oluşmaktadır. 
plt.hist(data.age,bins=30)
plt.xlabel("Yaş Değerleri")
plt.ylabel("Kaç Kişi")
plt.title("Test Edilen Yaş Grupları")
plt.show()
data.corr()

plt.figure(figsize=(12,6))
sns.heatmap(data[data.columns[0:]].corr(),annot=True)

#Grafiğe ve ısı haritasına baktığımız zaman korelasyonu en yüksek değerlerin başında age ve pregnant gelirken ikinci sırada triceps ve bmi geliyor. 
#Onları da insulin ve glucose değerleri izliyor.
#Yaş ve hamileliğin konumuzla bir alakası olduğunu düşünmediğimizden en yüksek korelasyonu triceps ve bmi arasında görüyoruz diyebiliriz.

fig, axes = plt.subplots(2,4, figsize = (16,8), sharex=False, sharey=False)
sns.boxplot(y='age',data=data, ax=axes[0,0])
sns.boxplot(y='pregnant',data=data, ax=axes[0,1])
sns.boxplot(y='glucose',data=data, ax=axes[0,2])
sns.boxplot(y='diabetes',data=data, ax=axes[0,3])
sns.boxplot(y='insulin',data=data, ax=axes[1,0])
sns.boxplot(y='triceps',data=data, ax=axes[1,1])
sns.boxplot(y='diastolic',data=data, ax=axes[1,2])
sns.boxplot(y='bmi',data=data, ax=axes[1,3])
plt.tight_layout()
# Yaş değerini genç seviyede olduğunu histogram grafiğinde de görmüştük. Burada optimal değer olarak glucose,triceps,diastolic, ve bmi yi gösterebiliriz.
# Ancak insulin miktarı çok aykırı değerlere sahiptir.

data.test = [1 if each == "positif" else 0 for each in data.test] 
#object türünde olan test feature sınıflandırmada kullanılamaz. Kategorik yada int olmak zorundadır. Bu yüzden bu şekilde int'e çevirdik
data.info()
#Sonuçlarımızın olduğu test feature nı datadan ayırıyoruz.
y = data.test.values
x_data = data.drop(["test"], axis=1)

#Normalizasyon işlemimizi yapıyoruz
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

#Train ve test edilecek dataları ayırıyoruz
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.15,random_state=42)


#LogisticRegression modelini kullanıp modeli eğitiyoruz
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
print("Test Doğruluğu(Accuracy) %{}".format(lr.score(x_test,y_test)*100))

y_pred = lr.predict(x_test)
y_true = y_test
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)

#Seaborn kullanarak confusion matriximizi heatmap aracıyla görselleştiriyoruz.
import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
predictions = lr.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))

# Bu modelin sonucuna baktığımız zaman çokda fena olmayan bir doğruluk değeri görüyoruz. 
# Şeker hastası olmayan hastaların tahmini iyi bir şekilde yapılmasına rağmen olanlarda ise pek verimli bir sonuç alınmamış

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
print("Test Doğruluğu(Accuracy) %{}".format(svc.score(x_test,y_test)*100))


y_pred = svc.predict(x_test)
y_true = y_test
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)

#Seaborn kullanarak confusion matriximizi heatmap aracıyla görselleştiriyoruz.

import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
predictions = svc.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))

# Bu modelin sonucuna baktığımız zaman LogisticRegression modeli ile birebir aynı sonuçları almış olduğumuzu görüyoruz.
# Diğer modelde olduğu gibi şeker hastası olmayan hastaların tahmini iyi bir şekilde yapılmasına rağmen olanlarda ise pek verimli bir sonuç alınmamış. 

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(3)
knn.fit(x_train,y_train)
print("Test Doğruluğu(Accuracy) %{}".format(knn.score(x_test,y_test)*100))

y_pred = knn.predict(x_test)
y_true = y_test
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)

#Seaborn kullanarak confusion matriximizi heatmap aracıyla görselleştiriyoruz.
import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
predictions = knn.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
# KNeighborsClassifier modelimizde diğer modellerden daha yüksek bir doğruluğa sahip bir rakam elde ettik.
# Doğruluk değerlerine baktığımızda ise negatif hastaların teşhisi konusunda bir tık daha kötü bir sonuç elde ederken pozitig hastalarda çok daha iyi sonuçlar vermiş durumda.
