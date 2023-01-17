# Kütüphaneler import ediliyor..
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import os
# Veriseti alındı..
print(os.listdir("../input"))
data = pd.read_csv('../input/column_2C_weka.csv')
# Verisetinin içeriğinden küçük bir kısım aşağıda görülmektedir.
data.head()
data.info()
data.describe()

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

# Daha sonra sınıflandırıcıları karşılaştrırken kullanılacak score değerlerini tutması için liste oluşturuldu.
SCORES = []
#%% KNN Modeli
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)

print("Accuracy of Naive Bayes algorithm: ",knn.score(x_test, y_test))

# Uygun k değerinin seçilmesi
score_list = []
for each in range(1,30):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train, y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1, 30), score_list)
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("Accuracy - k Value Relationship")
plt.show()

print("Max. Accurancy is ",np.max(score_list),"\nk value is :", score_list.index(np.max(score_list)))
SCORES.append(["KNN",np.max(score_list)])
# %% Naive bayes 
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
 
# test
acc_nb = nb.score(x_test,y_test)
print("Accuracy of Naive Bayes algorithm: ",acc_nb)
SCORES.append(["NB",acc_nb])

# %% SVM
 
from sklearn.svm import SVC
 
svm = SVC(random_state = 1)
svm.fit(x_train,y_train)
 
#  test
acc_svm = svm.score(x_test,y_test)
print("Accuracy of SVM algorithm: ",acc_svm)
SCORES.append(["SVM",acc_svm])
 
#%% Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

acc_dt = dt.score(x_test,y_test)
print("Accuracy of Decision Tree algorithm: ",acc_dt ) 
SCORES.append(["DT",acc_dt])
#%%  random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100,random_state = 1)
rf.fit(x_train,y_train)

acc_rf = rf.score(x_test,y_test)
print("Accuracy of Random Forest algorithm: ",acc_rf)
SCORES.append(["RF",acc_rf])
#%% Logistic Regression 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)

acc_lr = lr.score(x_test,y_test)
print("Accuracy of Logistic Regression algorithm: ",acc_lr)
SCORES.append(["LReg.",acc_lr])
# Sınıflandırıcıların Performanslarına bakıldığında SVM'in en yüksek başarıya sahip olduğunu
# görüyoruz.Decision Tree ise en düşük başarıya sahiptir.
SCORES = np.array(SCORES)
SCORES.sort(axis=0)

r,c = SCORES.shape
for idx in range(0, r):
    SCORES[idx,1] = '{:.5s}'.format(SCORES[idx,1])

plt.plot(SCORES[:,0],SCORES[:,1])
plt.ylabel("Accuracy")
plt.title("Accuracy - Classifiers Relationship")
plt.show()
