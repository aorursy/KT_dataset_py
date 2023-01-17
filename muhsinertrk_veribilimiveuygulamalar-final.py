import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, precision_score

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from collections import Counter
tablet = pd.read_csv("../input/tabletpc-priceclassification/tablet.csv")

df=tablet.copy()
#Veriyi tanıma,sindirme

#Veri dengelimi inceleme

#Korelasyon Matrisi
df.head()
df.info()
df.shape
df.corr()
df.describe().T
#Veri Görselleştirmeleri
#ısı haritasında Arka Kamere ve Ön kamare arasında güçlü bir ilişki var

#CozunurlukYükseklik ve CozunurlukGenislik arasında güçlü bir ilişki var
plt.subplots(figsize=(10, 10))

sns.heatmap(df.corr(), annot=True,cmap="YlGnBu")

plt.show()
df.hist(figsize = (15,15),color="purple")



plt.show()

#Tabletlerin ağırlıklarının,batarya gücünün,Cozunurluk Genisliklerinin,DahiliBelleklerin ve Ramlerin genel olarak dengeli dağıldığını görüyoruz.
sns.pairplot(df);
def countplot(baslik): 

    sns.countplot(x=baslik, data=df)

    plt.xticks(rotation=50)

    plt.show()

    

    print(df[baslik].value_counts())
countplot("FiyatAraligi")
countplot("Renk")
countplot("Bluetooth")
countplot("CiftHat")
countplot("4G")
countplot("3G")
countplot("Dokunmatik")
countplot("WiFi")
#Yukarıdaki grafiklerde gördüğümüz gibi sayısal olmayan verilerde 3G teknolojisi hariç stabil bir dağılım söz konusu
#Ortalama,Medyan ve Standart sapma
df.mean()#Verilerin ortalamalı ile medyanlarının çok yakın olduğunu görüyoruz yani veriler gayet stabil demektir.
df.std()

#Verilerin değerlerinin yakınlığını inceliyoruz.Standar sapmaları küçük olanların verileri birbirlerine daha yakın değerler içermektedir.
df.median()
#Ön işleme kısmı

#Eksik verlerin doldurulması

#Sayısallaşrılıma işlemleri
df.isna().sum()

#5 tane 4G,12 tane Ram verilerinde eksik var. Bu eksikler hesaplarımızda yanılmamıza sebep olabilir.Bir sonraki işlemde eksik verileri dolduracağız.
df.RAM = df.RAM.fillna(df.RAM.mean())#Eksik verileri ortalamaları ile dolduruyoruz.Çünkü verilerde sapmaya sebep olmamasını istiyoruz.
df.OnKameraMP = df.OnKameraMP.fillna(df.OnKameraMP.mean())
df.Bluetooth = df.Bluetooth.eq('Var').mul(1)

df.CiftHat = df.CiftHat.eq('Var').mul(1)

df["4G"] = df["4G"].eq('Var').mul(1)

df["3G"] = df["3G"].eq('Var').mul(1)

df.Dokunmatik = df.Dokunmatik.eq('Var').mul(1)

df.WiFi = df.WiFi.eq('Var').mul(1)
df.FiyatAraligi=pd.Categorical(df["FiyatAraligi"],ordered=True,categories=["Çok Ucuz","Ucuz","Normal","Pahalı"])
onehot_encoder = OneHotEncoder()
df.Renk = onehot_encoder.fit_transform(df[["Renk"]]).toarray()
df.Renk
#Makine Öğrenmesi
X = df.drop("FiyatAraligi",axis=1)

y= df["FiyatAraligi"]
X
y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
X_train
X_test
y_train
y_test
#LogisticRegression
logistic_regression=LogisticRegression()
logistic_model=logistic_regression.fit(X_train,y_train)
y_pred_LR=logistic_model.predict(X_test)
df_logistic = pd.DataFrame({"Gerçek Değerler" : y_test, "Tahmin Edilen" : y_pred_LR,"Tahmin Sonucu":(y_test==y_pred_LR)})



df_logistic
print("R Squared:", logistic_model.score(X_test, y_test))

print(classification_report(y_test, y_pred_LR))
LR_karmasiklik_matrisi = confusion_matrix(y_test, y_pred_LR)

print(LR_karmasiklik_matrisi)
#GaussianNB
Gaussian_NB= GaussianNB()
NB_model = Gaussian_NB.fit(X_train,y_train)
y_pred_NB=NB_model.predict(X_test)
df_NB = pd.DataFrame({"Gerçek Değerler" : y_test, "Tahmin Edilen" : y_pred_NB,"Tahmin Sonucu":(y_test==y_pred_NB)})



df_NB
print("R Squared:", NB_model.score(X_test, y_test))

print(classification_report(y_test, y_pred_NB))
DT_karmasiklik_matrisi = confusion_matrix(y_test, y_pred_NB)

print(DT_karmasiklik_matrisi)
#DecisionTree
DecisionTree = tree.DecisionTreeClassifier()
DecisionTree_model=DecisionTree.fit(X_train,y_train)
y_pred_DT=DecisionTree_model.predict(X_test)
df_DecisionTree = pd.DataFrame({"Gerçek Değerler" : y_test, "Tahmin Edilen" : y_pred_DT,"Tahmin Sonucu":(y_test==y_pred_DT)})



df_DecisionTree
print("R Squared:", DecisionTree_model.score(X_test, y_test))

print(classification_report(y_test, y_pred_DT))
DT_karmasiklik_matrisi = confusion_matrix(y_test, y_pred_DT)

print(DT_karmasiklik_matrisi)
#DecisionTree=>criterion='entropy'
DecisionTree_entropy = tree.DecisionTreeClassifier(criterion='entropy')
DecisionTree_entropy_model=DecisionTree_entropy.fit(X_train,y_train)
y_pred_DT_entropy=DecisionTree_entropy_model.predict(X_test)
df_DecisionTree_entropy = pd.DataFrame({"Gerçek Değerler" : y_test, "Tahmin Edilen" : y_pred_DT,"Tahmin Sonucu":(y_test==y_pred_DT_entropy)})



df_DecisionTree_entropy
print("R Squared:", DecisionTree_entropy_model.score(X_test, y_test))

print(classification_report(y_test, y_pred_DT_entropy))
DT_karmasiklik_matrisi_entropy = confusion_matrix(y_test, y_pred_DT_entropy)

print(DT_karmasiklik_matrisi_entropy)
#KNN
KNN = KNeighborsClassifier()
KNN_model=KNN.fit(X_train,y_train)
y_pred_KNN=KNN_model.predict(X_test)
df_KNN = pd.DataFrame({"Gerçek Değerler" : y_test, "Tahmin Edilen" : y_pred_KNN,"Tahmin Sonucu":(y_test==y_pred_KNN)})



df_KNN
print("R Squared:", KNN_model.score(X_test, y_test))

print(classification_report(y_test, y_pred_KNN))
KNN_karmasiklik_matrisi = confusion_matrix(y_test, y_pred_KNN)

print(KNN_karmasiklik_matrisi)
#KNN=>2,15
array = []

for i in range (2,15):

    KNN_n_neighbors = KNeighborsClassifier(n_neighbors=i)

    KNN_model_n_neighbors=KNN_n_neighbors.fit(X_train,y_train)

    y_pred_KNN_n_neighbors=KNN_model_n_neighbors.predict(X_test)

    print(i,".R Squared:", KNN_model_n_neighbors.score(X_test, y_test))

    KNN_score = KNN_model_n_neighbors.score(X_test, y_test)

    array.insert(i,KNN_score)
plt.plot(array)

plt.xlabel("k değerleri")

plt.ylabel("doğruluk skoru")
#KNN modeli 0.934 ile en yüksek başarı puanını veriyor.(n_neighbors=7 iken) daha sonra DT modeli 0.838 değeri geliyor ve en son GaussianNB modeli gelmekte