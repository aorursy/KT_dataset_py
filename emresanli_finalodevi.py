from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np 
import os 
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
import sklearn.tree as skltdtc
from sklearn.neighbors import KNeighborsClassifier
df1 = pd.read_csv('/kaggle/input/tablet.csv')
df1.dataframeName = 'tablet.csv'
df1.hist(layout=(5,4), figsize=(30,20))

plt.show()
def countplot(baslik): 
    sns.countplot(x=baslik, data=df1)
    plt.xticks(rotation=50)
    plt.show()
    
    print(df1[baslik].value_counts())
countplot("Bluetooth")
countplot("CiftHat")
countplot("4G")
countplot("3G")
countplot("WiFi")
countplot("Renk")
countplot("FiyatAraligi")
corr = df1.corr()
cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]

corr.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
    .set_caption("Kolerasyon Matrisinin Isı Haritası")\
    .set_precision(2)\
    .set_table_styles(magnify())
df1.info() # bu kod bloğunda datanın hakkında bilgi sahibi oluyoruz
df1.isnull().sum() # bu kod bloğunda datanın içinde toplamda 17 tane null değer olduğunu görüyoruz
print("BataryaGucu Ortalaması :",df1.BataryaGucu.mean())
print("BataryaGucu Medyanı :",df1.BataryaGucu.median())
print("BataryaGucu Standart Sapması :",df1.BataryaGucu.std())
print("Ortalama Medyana Çok Yakın")
print("Ortalama Standart Sapmadan Büyüktür.")
print("MikroislemciHizi Ortalaması :",df1.MikroislemciHizi.mean())
print("MikroislemciHizi Medyanı :",df1.MikroislemciHizi.median())
print("MikroislemciHizi Standart Sapması :",df1.MikroislemciHizi.std())
print("Ortalama Medyana Çok Yakın")
print("Ortalama Standart Sapmadan Büyüktür.")
print("OnKameraMP Ortalaması :",df1.OnKameraMP.mean())
print("OnKameraMP Medyanı :",df1.OnKameraMP.median())
print("OnKameraMP Standart Sapması :",df1.OnKameraMP.std())
print("Ortalama Medyana Yakın Değil")
print("Ortalama Standart Sapmaya Yakındır.")

print("DahiliBellek Ortalaması :",df1.DahiliBellek.mean())
print("DahiliBellek Medyanı :",df1.DahiliBellek.median())
print("DahiliBellek Standart Sapması :",df1.DahiliBellek.std())
print("Ortalama Medyana Çok Yakın")
print("Ortalama Standart Sapmadan Büyüktür.")

print("Kalinlik Ortalaması :",df1.Kalinlik.mean())
print("Kalinlik Medyanı :",df1.Kalinlik.median())
print("Kalinlik Standart Sapması :",df1.Kalinlik.std())
print("Ortalama Medyana Çok Yakın")
print("Ortalama Standart Sapmadan Büyüktür.")
print("Agirlik Ortalaması :",df1.Agirlik.mean())
print("Agirlik Medyanı :",df1.Agirlik.median())
print("Agirlik Standart Sapması :",df1.Agirlik.std())
print("Ortalama Medyana Çok Yakın")
print("Ortalama Standart Sapmadan Büyüktür.")

print("CekirdekSayisi Ortalaması :",df1.CekirdekSayisi.mean())
print("CekirdekSayisi Medyanı :",df1.CekirdekSayisi.median())
print("CekirdekSayisi Standart Sapması :",df1.CekirdekSayisi.std())
print("Ortalama Medyana Çok Yakın")
print("Ortalama Standart Sapmadan Büyüktür.")

print("ArkaKameraMP Ortalaması :",df1.ArkaKameraMP.mean())
print("ArkaKameraMP Medyanı :",df1.ArkaKameraMP.median())
print("ArkaKameraMP Standart Sapması :",df1.ArkaKameraMP.std())
print("Ortalama Medyana Çok Yakın")
print("Ortalama Standart Sapmadan Büyüktür.")
print("CozunurlukYükseklik Ortalaması :",df1.CozunurlukYükseklik.mean())
print("CozunurlukYükseklik Medyanı :",df1.CozunurlukYükseklik.median())
print("CozunurlukYükseklik Standart Sapması :",df1.CozunurlukYükseklik.std())
print("Ortalama Medyana Yakın Değil")
print("Ortalama Standart Sapmadan Büyüktür.")
print("CozunurlukGenislik Ortalaması :",df1.CozunurlukGenislik.mean())
print("CozunurlukGenislik Medyanı :",df1.CozunurlukGenislik.median())
print("CozunurlukGenislik Standart Sapması :",df1.CozunurlukGenislik.std())
print("Ortalama Medyana Çok Yakın")
print("Ortalama Standart Sapmadan Büyüktür.")
print("RAM Ortalaması :",df1.RAM.mean())
print("RAM Medyanı :",df1.RAM.median())
print("RAM Standart Sapması :",df1.RAM.std())
print("Ortalama Medyana Çok Yakın")
print("Ortalama Standart Sapmadan Büyüktür.")
df1.RAM = df1.RAM.fillna(df1.RAM.mean())
df1.OnKameraMP= df1.OnKameraMP.fillna(df1.OnKameraMP.mean())
df1.isnull().sum()
df1.Bluetooth = df1.Bluetooth.eq('Var').mul(1)
df1.CiftHat = df1.CiftHat.eq('Var').mul(1)
df1['4G'] = df1['4G'].eq('Var').mul(1)
df1['3G'] = df1['3G'].eq('Var').mul(1)
df1.WiFi = df1.WiFi.eq('Var').mul(1)
df1.Dokunmatik = df1.Dokunmatik.eq('Var').mul(1)
likert_scale = {'Pahalı':3, 'Normal':2, 'Ucuz':1, 'Çok Ucuz':0}
df1['FiyatAraligi'] = df1.FiyatAraligi.apply(lambda x: likert_scale[x])
renk_scale = {'Kırmızı':1 ,'Turkuaz':2, 'Gri':3, 'Turuncu':4, 'Kahverengi':5, 'Mor':6, 'Sarı':7, 'Yeşil':8, 'Siyah':9 ,'Mavi':10 , 'Beyaz':11,'Pembe':12}
df1['Renk'] = df1.Renk.apply(lambda x: renk_scale[x])
df1.sample(5)
Bagimsiz_degisken_x = df1.drop("FiyatAraligi",axis=1)
Bagimli_degisken_y= df1["FiyatAraligi"]
X_train, X_test, Y_train, Y_test = train_test_split(Bagimsiz_degisken_x,Bagimli_degisken_y, test_size = 1/4, random_state = 2, shuffle=1)
X_train
X_test
Y_train
Y_test
logmodel = LogisticRegression(solver = "liblinear") #  (solver = "liblinear") PARAMETRE UFAK VERİLER İÇİN DAHA TUTARLI OLUŞTUYOR
logmodel.fit(X_train,Y_train)
y_pred_logistic = logmodel.predict(X_test)
print(classification_report(Y_test,y_pred_logistic))
confusion_matrix = pd.crosstab(Y_test,y_pred_logistic, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)
NB = GaussianNB()
logmodel_NB = NB.fit(X_train,Y_train)
y_pred_NB = logmodel_NB.predict(X_test)
print(classification_report(Y_test,y_pred_NB))
confusion_matrix = pd.crosstab(Y_test,y_pred_NB, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)
tree = skltdtc.DecisionTreeClassifier()
logmodel_tree = tree.fit(X_train,Y_train)
y_pred_tree = logmodel_tree.predict(X_test)
print(classification_report(Y_test,y_pred_tree))
confusion_matrix = pd.crosstab(Y_test,y_pred_tree, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)
parametreli_tree = skltdtc.DecisionTreeClassifier(criterion = 'entropy')
logmodel_parametreli_tree = parametreli_tree.fit(X_train,Y_train)
y_pred_parametreli_tree = logmodel_parametreli_tree.predict(X_test)
print(classification_report(Y_test,y_pred_parametreli_tree))
confusion_matrix = pd.crosstab(Y_test,y_pred_parametreli_tree, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)
knn = KNeighborsClassifier()
logmodel_knn = knn.fit(X_train,Y_train)
y_pred_knn = logmodel_knn.predict(X_test)
print(classification_report(Y_test,y_pred_knn))
confusion_matrix = pd.crosstab(Y_test,y_pred_knn, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)
knn_array = []
for sayac in range(2,15):
    knn = KNeighborsClassifier(n_neighbors = sayac)
    logmodel_knn_sayacli = knn.fit(X_train,Y_train)
    y_pred_knn = logmodel_knn_sayacli.predict(X_test)
    print("KOMŞU SAYISI : " , sayac)
    print(logmodel_knn_sayacli.score(X_test,Y_test))
    knn_array.insert(sayac,logmodel_knn_sayacli.score(X_test,Y_test)) 

plt.plot(knn_array)
plt.xlabel("Komşu Sayısı - 2 ")
plt.ylabel("Başarı Oranı")
karsilastirma_array = []
karsilastirma_array.insert(1,"LogisticRegression = " + str(logmodel.score(X_test,Y_test)))
karsilastirma_array.insert(2,"GaussianNB = " + str(logmodel_NB.score(X_test,Y_test)))
karsilastirma_array.insert(3,"DecisionTree gini = " + str(logmodel_tree.score(X_test,Y_test)))
karsilastirma_array.insert(4,"DecisionTree entropy = " + str(logmodel_parametreli_tree.score(X_test,Y_test)))
karsilastirma_array.insert(5,"Parametresiz KNN = " + str(logmodel_knn.score(X_test,Y_test)))
for i in range(6,19):
    karsilastirma_array.insert(i,"KNN komşu sayısı " + str(i-4) + " =  " + str(knn_array[i-6]))
karsilastirma_array    