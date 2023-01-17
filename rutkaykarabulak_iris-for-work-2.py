# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Datasetimizi csv dosyasından okuyoruz.Datamız zambak bitkilerinin 3 türünü ele alıyor.
#Iris Setosa türü diğer İKİ Iris(Zambak) türünden tamamen ayrılırken,diğer İKİ tür ise benzerlikler göstermekte.
df = pd.read_csv("../input/Iris.csv")
#Veri Keşfi ve Görselleştirme BOLUM 1
#İlk 5 satırımızı alıyoruz.
df.head()
df = df.drop('Id', axis = 1) #ID özelliği bizim için gereksiz bir öznitelik siliyoruz...
df.head()
#Son 5 satırımızı alıyoruz.
df.tail()
#Rasal 5 satır
df.sample(5) 
#Bellek ve veri türleri
df.info()
#Satır sütun bilgisi
df.shape
df.describe() #Basit istatistikler
df.hist() #Histogram #SepalWith(çanak genişliği)'in ve SepalLenght'in daha mantıksal bit şekilde detaylandığını görüyoruz
#Korelasyon Gösterim
import seaborn as sns
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

#Sonuçlarımıza baktığımızda özniteliklerimizi incelersek;
#SepalLength(Çanak Uzunlugu) ve PetalLength(Taç yaprağı uzunluğu) ve PetalWidth arasında yüksek,iyi bir ilişki olduğundan,
#SepalWidth(Çanak Genişliği) ile diğer öznitelikler arasında hiç iyi bir ilişki olmadığından(0 ila 0.25 arası siyah bölge) 
#PetalLenght ile PetalWitdh,SepalLength arasında 0.75 1 arası iyi bir ilişki olduğundan 
# BAHSEDEBİLİRİZ...
#Eğer öznitelikler üzerinde işlem yapılacaksa öncelik olarak sepalLength ve petalLength üzerinde işlem yapmak mantıklı olacaktır.
#Korelasyon değeri yüksek olan iki öznitelik olan,SepalLength ve PetalLength için çizim işlemi yapalım

import matplotlib.pyplot as plt  
df.plot(x='SepalLengthCm', y='PetalLengthCm', style='o')
plt.title('SepalLengthCm-PetalLengthCm')
plt.xlabel('SepalLengthCm')  
plt.ylabel('PetalLengthCm')  
plt.show() 
#Güzel bir grafik elde ettiğimizi düşünüyorum.
#ÖN İŞLEME BOLUM 2

#1.Eksik değer kontrolü varsa doldurma ve yorumlama
df.isnull().sum()
#Null değerimiz bulunmamakta.

#Sınıf dağılımları nasıl?
df.groupby("Species").size()
#Peki güzel null değerimiz bulunmuyor... Peki ya null değerlerin olduğu bir dataset üzerinde çalışsaydık?
#1.Benzer şekilde null özellikler bulunacaktı --> df.isnull().sum()
#2.Toplam null değer içeren değerlerin sayisi bulunacaktı --> df.isnull().sum().sum()

#3.Bir eksik değer tablosu oluşturmak bizim için yararlı olacaktır.
#3:#Eksik değer tablosu
#def eksik_deger_tablosu(df): 
   #mis_val = df.isnull().sum()
    #mis_val_percent = 100 * df.isnull().sum()/len(df)
    #mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    #mis_val_table_ren_columns = mis_val_table.rename(
    #columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    #return mis_val_table_ren_columns
#4.Null değerleri kolon ismiyle beraber elimizle doldurabilirdik örnek olarak
#4: df['ÇanakYaprağı'] = df['ÇanakYaprağı'].fillna('Boş') -->Çanakyaprağının null değerlerini boş ile dolduruk.

#5.Herhangi bir kolondaki null değerleri eşsiz(unique) değerler ile doldurabilirdik
#5:print(df['Kolonİsmi'].unique()[IndisDegeri]) --> Indis değeri doldurmak istediğimiz değer(0,1,2,3..vs)
#df['Kolonİsmi'] = df['Kolonİsmi'].fillna(df['Kolonİsmi'].unique()[1]) --> Örneğin 1 ile doldurduk...

#6.Belli bir değerin üzerinde null değer içeren kolonları silebilirdik...
#6:#%70 üzerinde null değer içeren kolonları sil
#tr = len(df) * .3
#df.dropna(thresh = tr, axis = 1, inplace = True)
#2.Uç değerleri bulma ve yorumlama
df.describe()
import seaborn as sns
sns.boxplot(x=df['SepalLengthCm'])
#Alt uç değer ve üst uç değer hesaplaması yapalım.
#Describe'dan aldığımız %25 ve %75 değerleri üzerinde hesaplama yapalım.
#%75=6,4 ,%25=5,1
#Alt Uç Değer  = Q1 – 1.5(IQR) = 3,15
#Üst Uç Değer = Q3 + 1.5(IQR)= 8,35 IQR=1,3
P = np.percentile(df.SepalLengthCm, [1, 8]) 
P
new_df = df[(df.SepalLengthCm > P[0]) & (df.SepalLengthCm < P[1])]
new_df
df
#3.Mevcut özniteliklerden yeni bir öznitelik oluşturma
#Bir zambağın çanak yaprak oranı ve taç yaprak oranını bir öznitelik haline getirelim...
#Formülümüz uzunluk / genişlik olacak...
#
canak_Oran=(df['SepalLengthCm'] / df['SepalWidthCm'])
tac_Oran=(df['PetalLengthCm'] / df['PetalWidthCm'])

df['canakOran']=canak_Oran
df['tacOran']=tac_Oran
df
#4.Özniteliklerden bir tanesi (veya n tanesini) normalleştirme
#Veri Normalleştirme
from sklearn import preprocessing

#SepalLengthCm özniteliğini normalleştirmek istiyoruz.
x = df[['SepalLengthCm']].values.astype(float)

#Normalleştirme için MinMax normalleştirme yöntemini kullanıyoruz.
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df['SepalLengthCm2'] = pd.DataFrame(x_scaled)

df
#Veri Normalleştirme devam...
from sklearn import preprocessing

#SepalWidthCm özniteliğini normalleştirmek istiyoruz
x = df[['SepalWidthCm']].values.astype(float)

#Normalleştirme için MinMax normalleştirme yöntemini kullanıyoruz.
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df['SepalWidthCm2'] = pd.DataFrame(x_scaled)

df
df.corr() # Korelasyona bakalım...
#MODEL EGITIMI BOLUM 3
df.shape 
# Yeni eklediğimiz canakOran	tacOran	SepalLengthCm2	SepalWidthCm2 ile birlikte kolon sayımız 9 a yükseldi...
df.hist()
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection

#Eğitim için ilgili öznitlelik değerlerini seç,yeni kolonlar eklediğimiz için indislerin değişimine dikkat edelim...
X = df.iloc[:, :4].values

#Sınıflandırma öznitelik değerlerini seç
Y = df.iloc[:, 4].values
X
#Eğitim ve doğrulama veri kümelerinin ayrıştırılması
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#Naive Bayes modelinin oluşturulması
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
#NB modelinin K-katlamalı çapraz doğrulama ile ACC değerinin hesaplanması
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
cv_results 
#Modelimizin başarısı 1 değerlerinin çoğunluğu ve 0.91 0.83 gibi oranlarla gözler önüne serilmiştir.
#Sk katlamalı carpraz doğrulama ile çıkan sonuç matrisinin ortalaması ve standart sapmasının sonuçları
msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
msg

#Hatırlayalım standart sapmanın küçük olması homojen bir dağılım gösterdiğinin yani verilerin tutarlı olduğunun bir göstergesidir.
#Ortalamamızda 0.9750 olarak belirtilmiştir.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print("ACC: ",accuracy_score(y_pred,y_test))

#SONUCLARI YORUMLADIGIMIZDA 
                    #precision    recall  f1-score   support
 #   Iris-setosa       1.00      1.00      1.00        11
#Iris-versicolor       0.93      1.00      0.96        13
 #Iris-virginica       1.00      0.83      0.91         6
    #Tahminlerin 1 e yakın oluşu modelimizin başarısını temsil edecektir...
#Iris setora 1 lik precision,recall,f1-score ile modelimizin yaptığı tahminin başarısı görünmektedir.
#Iris-versicolor 0.93,1.00,0.96 ile gayet başarılı bir sonuc göstermekte.
#Aynı şekilde virginica ise 1.00 ve 0.83 0.91 lik sonuclarla modelimizin başarısı gözler önüne serilir.
#Prediction test in Doğruluğu:  0.9666666666666667 bizim için başarılı bir sonuç
#Karmaşıklık matrisi sonuçta belirtilmiştir...
#MODEL EĞİTİM DEVAM...
#Multiple Linear Regression yapalım...
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Verilerimizi değişkenlere atıyoruz
X = df.iloc[:,:4].values
y = df.iloc[:, 5].values
y # -- > kontrol amaçlı
#y değişkenini kontrol ettiğimizde elimizde string veriler olduğunu görüyoruz,bunlar algoritmamızın çalışması esnasında sayılar verilerle uyumluluk göstermesi adına sayısal verilere çevrilmeli
#Peki ne yapacağız ?
#Derste yaptığımız 50 startup örneğini hatırlayalım,şehirleri sayısal verilere çevirmiştik.
# Çözüm olarak Encoding yapıyoruz,datasetimizde string ifadelere sahip olduğumuz için bunları sayısal değerlere çevirmeye çalışacağız.

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
y
#Görüldüğü üzere Iris-virginica,Iris-setosa,Iris-versicolor türlerini 0,1,2 şeklinde sayısal değerlere çevirmiş olduk.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#datasetlerimizi model eğitimi için %20 olacak şekilde ayırdık
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)
df2 = pd.DataFrame({'Aslında olan': y_test, 'Tahmin Edilen': y_pred})  
df2
#Machine learning mantığıyla sonucunun ne olduğunu bildiğimiz fakat modelin tahmin etmesini istediğimiz durum göz önüne alınarak
#Actual ve Prediction adı altında yeni bir data frame oluşturarak modelimizin başarısını görüyoruz.
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
#Error oranlarımıza da bakarak eğitimimizi tamamlıyoruz...
#Model sonuçlarının karşılaştırılması ve yorumlanması BOLUM 4
#NB hakkında konusursak
#SONUCLARI YORUMLADIGIMIZDA 
                    #precision    recall  f1-score   support
 #   Iris-setosa       1.00      1.00      1.00        11
#Iris-versicolor       0.93      1.00      0.96        13
 #Iris-virginica       1.00      0.83      0.91         6
    #Tahminlerin 1 e yakın oluşu modelimizin başarısını temsil edecektir...
#Iris setora 1 lik precision,recall,f1-score ile modelimizin yaptığı tahminin başarısı görünmektedir.
#Iris-versicolor 0.93,1.00,0.96 ile gayet başarılı bir sonuc göstermekte.
#Aynı şekilde virginica ise 1.00 ve 0.83 0.91 lik sonuclarla modelimizin başarısı gözler önüne serilir.
#Prediction test in Doğruluğu:  0.9666666666666667 bizim için başarılı bir sonuç
#Karmaşıklık matrisi sonuçta belirtilmiştir...

# -->Görüldüğü gibi 1 değerine yakın olan tahminlerin fazla sayıda oluşu,recall ve f1-score'ların değerlerinin 1'e yakın oluşu datasetimizin NB karşısında gayet iyi bir sonuç verdiğini gösteriyor.
#Accuracy incelendiğinde ise 0.966 gibi oldukça başarılı bir sonuç elde edildi.

#NB'yi MulltipleLinearRegression ile karşılaştırdığımızda ise NB 1'e yakın değerlerin çokluğu recall precision f-1 score değerlerinin oldukça başarılı değerlere sahip olması NB'i bir adım daha öne çıkarıyor.
#NB daha başarılıdır...