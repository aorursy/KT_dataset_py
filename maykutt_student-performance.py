# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Model eğitimi ve başarım değerlendirmesi en sonda yer almaktadır.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Öğrenci Performans veri seti
df = pd.read_csv("../input/StudentsPerformance.csv")
df.describe() # temel istatistikler
df.info() # veri türleri ve bellek kullanımıyla ilgili bilgiler
df.head() # ilk 5 satır
df.tail() # son 5 satır
df.shape # satır, sütun sayısı
hist = df.hist(bins=3) # histogram çizdirme
df.isnull().sum() # boş değerleri kontrol etme
df.isnull().sum().sum() # boş değerleri kontrol etme
# Başarıyı etkilemeyen özniteliklerin tablodan çıkarılması
df = df.drop(columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'])
# Genel not ortalaması için Ortalama adında öznitelik oluşturulması
df['Ortalama'] = df.mean(numeric_only=True, axis=1)
# Başarı durumunu görmek için ortalama özniteliği kullanılarak Geçti özniteliği eklenmesi
def basari_durumu(puan):
    return (puan >= 70)

df['Geçti'] = df['Ortalama'].apply(basari_durumu)
df # çıkarılan ve yeni eklenen özniteliklerle tabloyu görme
df.corr() # korelasyon tablosu incelemesi
# Korelasyon ısı haritası gösterimi
import seaborn as sns
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
# Korelasyonu yüksek olan iki özniteliğin plotting işlemi
import matplotlib.pyplot as plt

df.plot(x='reading score', y='Ortalama', style='o')  
plt.title('reading score - Ortalama')  
plt.xlabel('reading score')  
plt.ylabel('Ortalama')  
plt.show() 
# Uç değerler tespiti
import seaborn as sns
sns.boxplot(x=df['Ortalama'])
# Uç değerler tespiti
P = np.percentile(df.Ortalama, [10, 100])
P
# Uç değerlerin çıkarılmış hali
new_df = df[(df.Ortalama > P[0]) & (df.Ortalama < P[1])]
new_df
# Veri normalleştirme
from sklearn import preprocessing

# Ortamala özniteliğinin normalleştirmesi
x = df[['Ortalama']].values.astype(float)

# Normalleştirme için MinMax normalleştirme yöntemini kullanıyor
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df['Ortalama2'] = pd.DataFrame(x_scaled)
df # Veri normalleştirme sonrası df
# Tahmin ve sınıf için kullanılacak özniteliklerin belirlenmesi
featuresForFit = ['math score', 'reading score', 'writing score']
X = df[featuresForFit]
y = df['Geçti']
# Tahmin ve sınıf için kullanılacak özniteliklerin belirlenmesi
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Karar ağaçları ile sınıflandırma ve ACC değerini yazdırma
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier().fit(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
# Karar ağacı için confusion matrix, precision, recall, f-measure gösterimi
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = clf.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
# Knn komşuluk ile sınıflandırma ve ACC değerini yazdırma
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))
# Knn komşuluk için confusion matrix, precision, recall, f-measure gösterimi
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
# Kullanılan ve sınıflama için düzenlenen 'Student Performance' verisetine uygulanan 'Karar ağacı' ve
# 'Knn komşuluk' sınıflandırmaları sonuçlarına göre elde edilen başarımlar yukarıda görüldüğü gibidir.
# Oluşturulan veri setinde sınıflandırma için 'Knn komşuluk' sınıflandırmasının başarımı daha yüksek
# olduğu için 'Knn komşuluk' sınıflandırması kullanılması daha doğru olacaktır.
#
# Veri setinde sınıflama yapmak için en yüksek doğruluğu sadece notlar kullanılarak alınabilirdi.
# 'Ortalama' özniteliği de X değişkeninde kullanıldığında doğruluk oranını daha fazla artırmaktadır.
