# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# datayi CSV veya baska kaynaklardan okutmak icin kullaniyoruz
import seaborn as sns # Korelasyon ısı haritası gösterimi için kullanıyoruz
import matplotlib.pyplot as plt # Korelasyonu yüksek olan iki özniteliğin plotting işlemi için kullanıyoruz

# sag tarafta draft environment bolumu var
# o bolumde input klasorumuze sistemden veya disardan ekledigimiz data dosyasi var
# yol tanimi ../input/train_distance_matrix.csv     seklinde

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
veriyolu = '../input/train_distance_matrix.csv' # dizini veriyolu olarak degiskene atadik 
data = pd.read_csv(veriyolu) #pandas'la read metoduyla yolunu yazdigimiz dosyayi aktardik 
data.describe() # bakalim neler aktarmisiz describe ile ilk bir kac satiri gorelim
data.columns  # veride hangi kolonlar var gorelim
# tum veri data degiskeninde oradan alacagimiz kolonlari tanimlayalim 
secilen_features = ['passenger_count', 'gc_distance', 'store_and_fwd_flag','trip_duration', 'google_distance', 'google_duration']
# simdi bu diziyi kullanarak X (tanimlayan kolonlar) ve Y (tahmin kolonu, price) sekline gecelim
# X degerlerini data dan elde edelim
X = data[secilen_features]
X.describe() # sectigimiz featurelar la veri kumesini gorelim
data.info() #bellek kullanımı ve veri türleri"
data.head() #ilk 5 satır"
data.tail() #son 5 satır  
data.shape # satır, sütun sayısı
hist = data.hist(bins=5) # histogram bakma
data.isnull().sum().sum() # boş değerlerin sayısını yazdırma
data.isnull().sum() # sutunlarda boş değerleri kontrol etme

data.corr() # korelasyon sonuçlarının tabloda görülmesi
#Korelasyon, Düz metin ve seaborn ısı haritası ile gösterilmesi
corr = data.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
# Korelasyonu yüksek olan iki tane öznitelik için plotting-çizim işlemi- gerçekleştrilmesi

data.plot(x='passenger_count', y='pickup_longitude', style='o')  
plt.title('passenger_count - pickup_longitude')  
plt.xlabel('passenger_count')  
plt.ylabel('pickup_longitude')  
plt.show() 
# Başarıyı etkilemeyen özniteliklerin tablodan çıkarılması
data = data.drop(columns = ['google_duration'])
data['durak'] = data.mean(numeric_only=True, axis=1)
# max. durak sayısını görmek için durak özniteliği kullanılarak max özniteliği eklenmesi
def max_durak(trip_duration):
    return (trip_duration >= 500)

data['max'] = data['durak'].apply(max_durak)
data # çıkarılan ve yeni eklenen özniteliklerle tabloyu görme
# Ön işleme//Uç değerleri bulma
import seaborn as sns
sns.boxplot(x=data['trip_duration'])
# Ön işleme//Uç değerleri bulma
import seaborn as sns
sns.boxplot(x=data['dropoff_latitude'])

# Ön işleme//Uç değerleri bulma
import seaborn as sns
sns.boxplot(x=data['pickup_longitude'])
# Ön işleme//Uç değerleri bulma
P = np.percentile(data.trip_duration, [10, 100])
P
# Uç değerlerin çıkarma
new_data = data[(data.trip_duration > P[0]) & (data.trip_duration < P[1])]
new_data
# Veri normalleştirme
from sklearn import preprocessing

# trip_duration özniteliğinin normalleştirmesi
x = data[['trip_duration']].values.astype(float)

# Normalleştirme için MinMax normalleştirme yöntemini kullanıyor
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data['trip_duration2'] = pd.DataFrame(x_scaled)
data # Veri normalleştirme sonrası data
# Tahmin ve sınıf için kullanılacak özniteliklerin belirlenmesi
featuresForFit = ['dropoff_latitude', 'pickup_latitude']
X = data[featuresForFit]
y = data['max']
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