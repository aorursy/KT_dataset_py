import pandas as pd               # dataframe manipülasyon işlemleri için kullanacağız.

import numpy as np                # vektörel ve matris işlemleri için kullanacağız.

import seaborn as sns             # görselleştirme yapmak için kullanacağız.

import matplotlib.pyplot as plt



import missingno                  # eksik verileri daha iyi okumak için kullanacağız.

from sklearn import preprocessing   # ön işleme aşamasında label encoding vb. için dahil ettik.

import re                         # regular expression yani düzenli ifadeler kullanmak için dahil ettik.



#train test modellerini oluştumak için gerekli kütüphaneler.

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, precision_score

from sklearn.naive_bayes import GaussianNB



from sklearn.preprocessing import scale 

from sklearn import model_selection

from sklearn.tree import DecisionTreeRegressor



from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import BaggingRegressor

from argparse import Namespace







from sklearn.neighbors import KNeighborsClassifier

from sklearn import tree

from sklearn.metrics import recall_score, f1_score, precision_score

from sklearn.tree import DecisionTreeClassifier





from sklearn import ensemble

from sklearn.metrics import confusion_matrix as cm

from matplotlib.legend_handler import HandlerLine2D





from sklearn import preprocessing

from warnings import filterwarnings

filterwarnings('ignore')

tablet = pd.read_csv("../input/tablet.csv")  

df = tablet.copy()
df.head(8)
df.info()
df.shape 

df.isna().sum() 
df.describe().T
sns.scatterplot(x = "ArkaKameraMP", y = "OnKameraMP", data = df ); 
df.corr()
sns.heatmap(df.corr());
sns.scatterplot(x = "ArkaKameraMP", y = "OnKameraMP", data = df ); 
df.hist(figsize =(20,17),bins=18)

plt.show()
sns.boxplot(df["OnKameraMP"])
df["FiyatAraligi"].unique()
missingno.matrix(df,figsize=(20,12));
missingno.heatmap(df, figsize= (8,5)); 
def eksik_deger_tablosu(df): 

    eksik_deger = df.isnull().sum()

    eksik_deger_yuzde = 100 * df.isnull().sum()/len(df)

    eksik_deger_tablo = pd.concat([eksik_deger, eksik_deger_yuzde], axis=1)

    eksik_deger_tablo_son = eksik_deger_tablo.rename(columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})

    return eksik_deger_tablo_son
eksik_deger_tablosu(df)
df["RAM"].head(15)
df.groupby("FiyatAraligi").mean()
df.groupby("FiyatAraligi")[["RAM"]].mean()
df[(df["FiyatAraligi"] == "Normal") & (df["RAM"].isnull())]
df[(df["FiyatAraligi"] == "Pahalı") & (df["RAM"].isnull())] 
null_RAM_Pahalı = df[(df["FiyatAraligi"] == "Pahalı") & (df["RAM"].isnull())].index

null_RAM_Pahalı
df.loc[null_RAM_Pahalı ,"RAM"] = 3449.35041 
df.isna().sum()["RAM"]
df["OnKameraMP"].unique()
df.groupby("ArkaKameraMP").mean()
df.groupby("ArkaKameraMP")[["OnKameraMP"]].mean()
df[(df["ArkaKameraMP"] == 0) & (df["OnKameraMP"].isnull())]
df[(df["ArkaKameraMP"] == 3) & (df["OnKameraMP"].isnull())]
df[(df["ArkaKameraMP"] == 9) & (df["OnKameraMP"].isnull())]
df[(df["ArkaKameraMP"] == 14) & (df["OnKameraMP"].isnull())]
df[(df["ArkaKameraMP"] == 20) & (df["OnKameraMP"].isnull())]
null_OnKameraMP_0 = df[(df["ArkaKameraMP"] == 0 ) & (df["OnKameraMP"].isnull())].index

null_OnKameraMP_3 = df[(df["ArkaKameraMP"] == 3 ) & (df["OnKameraMP"].isnull())].index

null_OnKameraMP_9 = df[(df["ArkaKameraMP"] == 9 ) & (df["OnKameraMP"].isnull())].index

null_OnKameraMP_14 = df[(df["ArkaKameraMP"] == 14 ) & (df["OnKameraMP"].isnull())].index

null_OnKameraMP_20 = df[(df["ArkaKameraMP"] == 20 ) & (df["OnKameraMP"].isnull())].index
df.loc[null_OnKameraMP_0 ,"OnKameraMP"] = 0

df.loc[null_OnKameraMP_3 ,"OnKameraMP"] = 1

df.loc[null_OnKameraMP_9 ,"OnKameraMP"] = 4

df.loc[null_OnKameraMP_14 ,"OnKameraMP"] = 7

df.loc[null_OnKameraMP_20 ,"OnKameraMP"] = 9
df.isna().sum()["OnKameraMP"]
df["OnKameraMP"].unique()
df.isna().sum()
label_encoder = preprocessing.LabelEncoder()
df["Fiyat_Encoded"]= label_encoder.fit_transform(df['FiyatAraligi'])

Fiyat_Encoded = df[(df["FiyatAraligi"] == "Çok Ucuz" )].index

df.loc[Fiyat_Encoded ,"Fiyat_Encoded"] = 1 
Fiyat_Encoded = df[(df["FiyatAraligi"] == "Ucuz" )].index

df.loc[Fiyat_Encoded ,"Fiyat_Encoded"] = 2 
Fiyat_Encoded = df[(df["FiyatAraligi"] == "Normal" )].index

df.loc[Fiyat_Encoded ,"Fiyat_Encoded"] = 3
Fiyat_Encoded = df[(df["FiyatAraligi"] == "Pahalı" )].index

df.loc[Fiyat_Encoded ,"Fiyat_Encoded"] = 4 
df.head(5)
Bluetooth = df[(df["Bluetooth"] == "Yok" )].index

df.loc[Bluetooth ,"Bluetooth"] = 0 

Bluetooth = df[(df["Bluetooth"] == "Var" )].index

df.loc[Bluetooth ,"Bluetooth"] = 1 
df.head()
CiftHat = df[(df["CiftHat"] == "Yok" )].index

df.loc[CiftHat ,"CiftHat"] = 0 

CiftHat = df[(df["CiftHat"] == "Var" )].index

df.loc[CiftHat ,"CiftHat"] = 1 

df.head()
df['4G'] = label_encoder.fit_transform(df['4G']).T
df.head()
df['3G'] = label_encoder.fit_transform(df['3G'])
df.head()
Dokunmatik = df[(df["Dokunmatik"] == "Yok" )].index

df.loc[Dokunmatik ,"Dokunmatik"] = 0 

Dokunmatik = df[(df["Dokunmatik"] == "Var" )].index

df.loc[Dokunmatik ,"Dokunmatik"] = 1 
df.head() # Kontrol ediyoruz.
WiFi = df[(df["WiFi"] == "Yok" )].index

df.loc[WiFi ,"WiFi"] = 0 

WiFi = df[(df["WiFi"] == "Var" )].index

df.loc[WiFi ,"WiFi"] = 1 
df.head()
df['Renk'].unique()
df.groupby("Renk").mean()
df.drop(["Renk"], axis = 1, inplace = True)
df.head()
sns.heatmap(df.corr());
df.groupby("4G")[["4G"]].mean()
sns.distplot(df["RAM"], bins=16, color="green");
sns.scatterplot(x = "RAM", y = "Fiyat_Encoded", data = df);
sns.jointplot(x = df["Fiyat_Encoded"], y = df["RAM"], kind = "kde", color = "purple");
sns.catplot(x = "Fiyat_Encoded", y = "RAM", data = df); 
BagimliD = df["Fiyat_Encoded"] # Bağımlı değişken(hedef değişken)

BagimsizD = df.drop(["FiyatAraligi"] , axis = 1)

BagimsizD = BagimsizD.drop(["Fiyat_Encoded"] , axis = 1) # Bağımsız değişkenler listesi
BagimliD.head(8) 
BagimsizD.head(8)
BagimsizD_train, BagimsizD_test, BagimliD_train, BagimliD_test = train_test_split(BagimsizD, BagimliD, test_size = 0.25 , train_size = 0.75 , random_state = 0, shuffle=1)
BagimsizD_train.head()
BagimsizD_test.head()
BagimliD_train.head()
BagimliD_test.head()
nb = GaussianNB()

nb_model = nb.fit(BagimsizD_train, BagimliD_train)
nb_model
dir(nb_model)
BagimsizD_test[0:10]
nb_model.predict(BagimsizD_test)[0:10]
BagimliD_test[0:10]
BagimliD_pred = nb_model.predict(BagimsizD_test)
BagimliD_pred
BagimliD_test
accuracy_score(BagimliD_test, BagimliD_pred)
karmasiklik_matrisi = confusion_matrix(BagimliD_test, BagimliD_pred)

print(karmasiklik_matrisi)
(karmasiklik_matrisi[0][0] + karmasiklik_matrisi[1][1]+ karmasiklik_matrisi[2][2]+ karmasiklik_matrisi[3][3]) / (karmasiklik_matrisi[0][0] +karmasiklik_matrisi[0][1] + karmasiklik_matrisi[0][2]+ karmasiklik_matrisi[0][3] + karmasiklik_matrisi[1][0] + karmasiklik_matrisi[1][1] + karmasiklik_matrisi[1][2]+ karmasiklik_matrisi[1][3] + karmasiklik_matrisi[2][0]+ karmasiklik_matrisi[2][1] + karmasiklik_matrisi[2][2]+ karmasiklik_matrisi[2][3] + karmasiklik_matrisi[3][0]+ karmasiklik_matrisi[3][1] + karmasiklik_matrisi[3][2]+ karmasiklik_matrisi[3][3])
PrecisionScore = precision_score(BagimliD_test, BagimliD_pred, average='weighted')

PrecisionScore
RecallScore = recall_score(BagimliD_test, BagimliD_pred, average='weighted')

RecallScore
F1Score = f1_score(BagimliD_test, BagimliD_pred, average = 'weighted')  

F1Score
print(classification_report(BagimliD_test, BagimliD_pred))
cart = DecisionTreeClassifier(random_state = 42, criterion='entropy')

cart_model = cart.fit(BagimsizD_train, BagimliD_train)
cart_model
cart_model
df.columns
BagimliD_pred = cart_model.predict(BagimsizD_test)
accuracy_score(BagimliD_test, BagimliD_pred)
karmasiklik_matrisi = confusion_matrix(BagimliD_test, BagimliD_pred)

print(karmasiklik_matrisi)
print(classification_report(BagimliD_test, BagimliD_pred))
from sklearn.tree import export_graphviz

from sklearn import tree

from IPython.display import SVG

from graphviz import Source

from IPython.display import display

graph = Source(tree.export_graphviz(cart, out_file = None, feature_names = BagimsizD.columns, filled = True))

ranking = cart.feature_importances_

features = np.argsort(ranking)[::-1][:10]

columns = BagimsizD.columns



plt.figure(figsize = (15, 9))

plt.title("Karar Ağacına Göre Özniteliklerin Önem Derecesi", y = 1.03, size = 18)

plt.bar(range(len(features)), ranking[features], color="lime", align="center")

plt.xticks(range(len(features)), columns[features], rotation=80)

plt.show()
knn = KNeighborsClassifier()

knn_model = knn.fit(BagimsizD_train, BagimliD_train)
knn_model
BagimliD_pred = knn_model.predict(BagimsizD)
accuracy_score(BagimliD, BagimliD_pred)
karmasiklik_matrisi = confusion_matrix(BagimliD, BagimliD_pred)

print(karmasiklik_matrisi)
print(classification_report(BagimliD, BagimliD_pred))
knn_params = {"n_neighbors": np.arange(2,15)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, knn_params, cv = 3)

knn_cv.fit(BagimsizD_train, BagimliD_train)
print("En iyi skor: " + str(knn_cv.best_score_))

print("En iyi parametreler: " + str(knn_cv.best_params_))
score_list = []



for each in range(2,15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(BagimsizD_train,BagimliD_train)

    score_list.append(knn2.score(BagimsizD_test, BagimliD_test))



plt.plot(range(2,15),score_list)

plt.xlabel("k en yakın komşu sayıları")

plt.ylabel("doğruluk skoru")

plt.show()