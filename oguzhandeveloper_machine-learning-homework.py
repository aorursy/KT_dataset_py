#Kütüphanelerin eklenmesi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Dataset'in eklenemesi ve DataFrame veri yapısına dönüştürme
dataPath = "../input/abalone.csv"
data = pd.read_csv(dataPath)
dataColumns = data.columns

df = pd.DataFrame(data, columns = dataColumns)
#DataFrame'in ilk 5 örneği
df.head()
#DataFrame'in satır ve kolon sayıları
df.shape
#DataFrame'in örneklerine ait matematiksel değerlendirmeleri
df.describe()
#DataFrame'in kolon bazlı incelenmesi (toplam satır ve kolonun veri yapısı)
df.info()
#DataFrame'in içerisinde eksik veri var mı?
df.isnull().sum()
#DataFrame 'Rings'(yaş) aralıklarına göre nasıl dağılmış?
df.groupby("Rings").size()
#Accuracy'i arttırmak için 'Rings' (Yaş) değerlerinde en az olanların bulunduğu
#aralığının ( 12 - 29 yaş aralığı) silinmesi Note: Zorunlu değildir
df = df.drop(df[(df['Rings']>=12)].index)
#Satır sayısına ve kolon sayısına tekrar bakıyoruz
df.shape
#Tekrar bakıyoruz
df.groupby("Rings").size()
#'Sex' niteliği hariç tüm değerler sayıdır(int). 'Sex' niteliğini sayısal değerlere çevirmemiz gerekir
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

#df['Sex'] = lb.fit_transform(df['Sex'])
#DataFrame'in histogram grafiğine bakıyoruz
df.hist()
#Kutu çizim grafiği incelemesi
df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
df.corr()
#Gördüğümüz gibi 'Sex' özelliğini korelasyonu kötüdür. Bu özelliği(kolonu) kaldırmalıyız
#Ek olarak 'Shucked weight','Viscera weight','Whole weight' nitelikleri 'Rings' niteliği ile
#korelasyonu en düşük olanlardır. Accuracy' i arttırmak için bu kolonlarda silinebilir.
df = df.drop(['Sex'], axis = 1)
df = df.drop(['Shucked weight','Viscera weight','Whole weight'], axis = 1)
#Toplam satır sayısı ve kolon sayısı
df.shape
#Kütüphaneleri eklenmesi
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection
import random

#Eğitim için ilgili öznitlelik değerlerini seç
X = df.iloc[:, :-1].values

#Sınıflandırma öznitelik değerlerini seç
Y = df.iloc[:, -1].values
#Eğitim ve doğrulama veri kümelerinin ayrıştırılması
validation_size = 0.20
seed = random.randint(1,200)
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#Naive Bayes modelinin oluşturulması
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
#NB modelinin K-katlamalı çapraz doğrulama ile ACC değerinin hesaplanması
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#Çıkan sonuçların kontrolu
cv_results
#çıkan sonucun ortalamasına ve standart sapmasına bakıyoruz
msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
msg
#modeli eğitmemiz gereken test ve eğitim verileri için kütüphane eklenmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#modelin eğitimi
model.fit(X_train, y_train)

#modelin tahmin edilmesi
y_pred = model.predict(X_test)

# Sınıflandırıcı tarafından yapılan tahminlerin özeti
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score (Doğruluk Skoru)
from sklearn.metrics import accuracy_score
print("ACC: ",accuracy_score(y_pred,y_test))
#DataFrame veri yapısı tekrar oluşturuluyor
df = pd.DataFrame(data, columns = dataColumns)
#Tekrar 'Sex' niteliği çıkarılıyor
df = df.drop(['Sex'], axis = 1)
X = df.iloc[:,:-1].values
y = df.iloc[:, 7].values
#Korelasyonlar kontrol ediliyor
df.corr()
#'Height' nitelikleri için 'Rings' niteliğinin düzlemde dağılımı inceleniyor
df.plot(x='Height', y='Rings', style='o')  
plt.title('Height - Rings')  
plt.xlabel('Height')  
plt.ylabel('Rings')  
plt.show() 
#Histogram grafiklerine bakılıyor
df.hist()
#X kontrol ediliyor
X
#Modeli eğitmek ve test etmek için dataframe veri yapısını bölmeliyiz
#Kütüphanelerin eklenmesi
from sklearn.model_selection import train_test_split

#Modelin test ve eğitim olarak ayrılması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#Modelin eğtilmesi için LinearRegresyon modeli ekleniyor
from sklearn.linear_model import LinearRegression
model = LinearRegression()
#Model eğitiliyor
model.fit(X_train, y_train)

# test kümesi ile tahmin yaılıyor
y_pred = model.predict(X_test)
#Tahmin ve Gerçek Değerlerin Karşılaştırması
df2 = pd.DataFrame({'Gerçek Değer': y_test, 'Tahmin Edilen Değer': y_pred})  
df2.head()
#Hata oranlarının gösterilmesi
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
