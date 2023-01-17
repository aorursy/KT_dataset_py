# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# %% Veriseti alındı.
data = pd.read_csv('../input/column_3C_weka.csv')
# Verisetinin içeriğinden küçük bir kısım aşağıda görülmektedir.
data.head()


#%% Veri sayısal olarak ifade ediliyor.

data.loc[:,'class'] = [1 if each == 'Normal' else 0 for each in data.loc[:,'class'] ]
Labels = data.loc[:,'class']

x = data.drop(["class"],axis = 1)
#%% Normalizasyon..
x_norm = (x - np.min(x))/(np.max(x) - np.min(x))

print("NORMALİZASYON İŞLEMİ ÖNCESİ:",
      "\nMin :")
print(np.min(x))
print("\nMax :")
print(np.max(x))


print("\n\nNORMALİZASYON İŞLEMİ SONRASI:",
      "\nMin :")
print(np.min(x_norm))
print("\nMax :")
print(np.max(x_norm))


#%% Veri Eğitim ve Test verisi olarak ayrıldı..
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_norm, Labels, test_size = 0.3, random_state = 1)
# %% Buradaki fonksiyon 2 nokta arasındaki uzaklığı hesaplamaktadır.
# Distance = Sqrt(Sum((p1-p2)^2)) 
#
def Distance(point_1,point_2):
    total = 0
    for idx in range(len(point_1)):
        total = total + (point_1[idx] - point_2[idx])**2
    return total**0.5
    
    
def K_NNeighbors(k_value, x_train, y_train, x_test):
    y_predict = []
    
    #Herbir test noktası için diğer tüm noktalara olan uzaklıklar hesaplanıyor.
    #Bulunan uzaklıklar etiketlerle beraber 'Neighbors' değişkeninde tutuluyor.
    for idx_test in range(x_test.shape[0]):
        Neighbors = []
        test_point = x_test[idx_test]
        for idx_rows in range(x_train.shape[0]):
            train_point = x_train[idx_rows]
            Neighbors.append([Distance(test_point, train_point),y_train[idx_rows]])
        
        # Her bir komşunun test noktasına olan uzaklığı bulunuyor.En yakın 'K' tane komşuyu seçmek için 
        # öncelikle komşular uzaklıklarına göre küçükten büyüğe doğru sıralanıyor..
        # Daha sonra k tane komşu seçilip içerisinden etiket(label) değerleri çekiliyor.
        Neighbors.sort()
        Neighbors = Neighbors[0:k_value]
        Labels = [n[1] for n in Neighbors]
        
        # En yakın k tane komşunun sahip olduğu etiketlerin frekansları bulunuyor ve en yüksek frekansa sahip
        # etiket test noktasını sınıflamakta kullanılıyor.
        from itertools import groupby
        Freq = [[len(list(group)), key] for key, group in groupby(Labels)]
        y_predict.append(max(Freq)[1])
    return y_predict
        
            
    
    
# Yazılan KNN algoritması deneniyor.
y_predicted = K_NNeighbors(5, np.array(x_train), np.array(y_train), np.array(x_test))
# Yazılan algoritmanın doğruluğu ölçülüyor.
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_predicted))
#%% KNN Modeli
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)

print("Accuracy of KNN algorithm: ",knn.score(x_test, y_test))
# Algoritmaların Karşılaştırılması.
score_list_sklearn = []
score_list_myknn = []

for each in range(1,50):
    sklearn_knn = KNeighborsClassifier(n_neighbors=each)
    sklearn_knn.fit(x_train, y_train)
    
    y_predicted = K_NNeighbors(each, np.array(x_train), np.array(y_train), np.array(x_test))
    
    score_list_myknn.append(accuracy_score(y_test, y_predicted))
    score_list_sklearn.append(sklearn_knn.score(x_test,y_test))
 
plt.plot(range(1, 50), score_list_sklearn)
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("KNN With Sklearn")
plt.show()

plt.plot(range(1, 50), score_list_myknn)
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("My KNN")
plt.show()