import pandas as pd

import numpy as np

import seaborn as sns

import missingno   

from sklearn import preprocessing 
df = pd.read_csv("../input/tabletpc-priceclassification/tablet.csv")
df.head()
df.shape
df.dtypes
df.info()
df.isna().sum()
df.isna().sum().sum()
df.describe().T
df.corr()
corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
df["Bluetooth"].unique() # Sadece iki adet benzersiz değişkenimiz olduğunu görüyoruz.
df["OnKameraMP"].nunique() # on kameramızın benzersiz değişkenleri bluetooth ile kıyasladığımızda oldukça fazla.
df["FiyatAraligi"].unique() # 5 adet fiyat benzersiz değişkenimiz mevcut. Son olarak renklere de göz atalım.
df["Renk"].unique() # renk tarafından da zengin bir içeriğe sahip olduğumuzu söylemek yanlış olmaz. 
df["Renk"].value_counts() # Renk dağılımı çok da dengeli değil.
df["WiFi"].value_counts() # Wifi özniteliğimizin dağılımına baktığımızda neredeyse dengeli olduğunu söyleyebiliriz.
df["BataryaOmru"].value_counts() # Batarya ömürlerine baktığımızda ise bizi biraz daha farklı bir dağılım karşılıyor.
sns.countplot(x = "BataryaOmru", data = df)
sns.countplot(x = "Renk", data = df)
sns.countplot(x = "FiyatAraligi", data = df)
sns.countplot(x = "DahiliBellek", data = df)
sns.countplot(x = "Agirlik", data = df)
sns.distplot(df["BataryaGucu"], bins = 16, color = "red");
sns.distplot(df["DahiliBellek"], bins = 16, color = "red");
sns.distplot(df["OnKameraMP"], bins = 16, color = "red");
sns.jointplot(x = "ArkaKameraMP", y = "OnKameraMP", data = df, color = "purple");
sns.jointplot(x = "ArkaKameraMP", y = "OnKameraMP", data = df, color = "blue", kind = "kde");
sns.barplot(x = "FiyatAraligi", y = "RAM", data = df);
sns.barplot(x = "FiyatAraligi", y = "CekirdekSayisi", data = df);
sns.barplot(x = "WiFi", y = "BataryaGucu", data = df);
sns.barplot(x = "ArkaKameraMP", y = "BataryaGucu", data = df);
df.corr()["ArkaKameraMP"]["DahiliBellek"]
sns.scatterplot(x = "ArkaKameraMP", y = "DahiliBellek", data = df, color="purple");
sns.lmplot(x = "CozunurlukYükseklik", y = "CozunurlukGenislik", data = df, hue = "Dokunmatik");
sns.violinplot(y = "DahiliBellek", data = df);
sns.violinplot(x = "4G", y = "BataryaOmru", data = df);
sns.violinplot(x = "Bluetooth", y = "BataryaOmru", data = df);
sns.scatterplot(x = "RAM", y = "BataryaGucu",hue = "FiyatAraligi",data = df);
sns.jointplot(x = "Kalinlik", y = "Agirlik", data = df, kind = "kde")
sns.barplot(x ="FiyatAraligi" , y = "Kalinlik" , data = df);
sns.barplot(x ="FiyatAraligi" , y = "ArkaKameraMP" , data = df);
df.sample(5)
sns.factorplot("FiyatAraligi", "BataryaGucu", "CiftHat", data = df, kind = "bar");
sns.factorplot("FiyatAraligi", "RAM", "Renk", data = df, kind = "bar");
missingno.matrix(df,figsize=(20, 10));
df.isnull().sum().sum()
missingno.heatmap(df, figsize= (20,8));
def eksik_deger_tablosu(df): 

    eksik_deger = df.isnull().sum()

    eksik_deger_yuzde = 100 * df.isnull().sum()/len(df)

    eksik_deger_tablo = pd.concat([eksik_deger, eksik_deger_yuzde], axis=1)

    eksik_deger_tablo_son = eksik_deger_tablo.rename(columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})

    return eksik_deger_tablo_son
eksik_deger_tablosu(df)
df["RAM"].unique() # Acaba kaç tane? 
df["RAM"].nunique() # 1554 adet benzersiz değerimiz var. Ram miktarının fiyatlandırmayı etkileyip etkilemediğine tekrar bakalım.
sns.barplot(x = "FiyatAraligi", y = "RAM", data = df);
df["OnKameraMP"].unique() # Acaba kaç tane? 
df["OnKameraMP"].nunique() # Ram özniteliğine kıyasla oldukça az benzersiz değere sahip olduğunu görüyoruz.
sns.barplot(x = "FiyatAraligi", y = "OnKameraMP", data = df);
sns.countplot(df[df["RAM"].isnull()]["FiyatAraligi"]);
sns.countplot(df[df["OnKameraMP"].isnull()]["FiyatAraligi"]);
df[(df["OnKameraMP"].isnull())]
kamerasi_gercekten_eksik_olanlar = df[(df["4G"] == "Var") & (df["OnKameraMP"].isnull())].index

kamerasi_gercekten_eksik_olanlar
df.groupby("FiyatAraligi").mean()
df.loc[kamerasi_gercekten_eksik_olanlar,"OnKameraMP"] = 4.01
kamera_verisi_degisecekler = df[(df["4G"] == "Yok") & (df["OnKameraMP"].isnull())].index

kamera_verisi_degisecekler
df.loc[kamera_verisi_degisecekler,"OnKameraMP"] = 0
kamera_null_kontrolu = df[(df["4G"] == "Var") & (df["OnKameraMP"].isnull())].index

kamera_null_kontrolu
ram_degeri_null_olanlar = df[(df["RAM"].isnull())].index

ram_degeri_null_olanlar
df.groupby("FiyatAraligi").mean()
df.loc[ram_degeri_null_olanlar,"RAM"] = 3449 ## Gerekli atamayı yapıyoruz. Ardından tekrardan kontrol edelim.
ram_degeri_null_kontrol = df[(df["RAM"].isnull())].index

ram_degeri_null_kontrol 
df.isnull().sum().sum()
o_kontrolu = df[(df["BataryaGucu"] == 0)]

o_kontrolu
o_kontrolu = df[(df["MikroislemciHizi"] == 0)]

o_kontrolu
o_kontrolu = df[(df["OnKameraMP"] == 0)]

o_kontrolu.head(10)
o_kontrolu_on_pahali = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Pahalı")].index

o_kontrolu_on_pahali
df.groupby("FiyatAraligi").mean()
df.loc[o_kontrolu_on_pahali,"OnKameraMP"] = 4.31
o_kontrolu_on_pahali = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Pahalı")].index

o_kontrolu_on_pahali
o_kontrolu_on_normal = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Normal")].index

o_kontrolu_on_normal
df.groupby("FiyatAraligi").mean()
df.loc[o_kontrolu_on_normal,"OnKameraMP"] = 4.5
o_kontrolu_on_normal = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Normal")].index

o_kontrolu_on_normal
o_kontrolu_on_ucuz = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Ucuz")]

o_kontrolu_on_ucuz
df.groupby("FiyatAraligi").mean()
o_kontrolu_on_ucuz = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Ucuz") & (df["RAM"] > 1500)].index

o_kontrolu_on_ucuz
df.loc[o_kontrolu_on_ucuz,"OnKameraMP"] = 4.34
o_kontrolu_on_ucuz = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Ucuz") & (df["RAM"] > 1500)].index

o_kontrolu_on_ucuz
o_kontrolu_on_ucuz = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Ucuz")]

o_kontrolu_on_ucuz
o_kontrolu_on_ucuz = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Ucuz")& (df["WiFi"] == "Var")]

o_kontrolu_on_ucuz
o_kontrolu_on_ucuz = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Ucuz")& (df["WiFi"] == "Var")].index

o_kontrolu_on_ucuz
df.groupby("FiyatAraligi").mean()
df.loc[o_kontrolu_on_ucuz,"OnKameraMP"] = 4.95
o_kontrolu_on_ucuz = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Ucuz") & (df["WiFi"] == "Var")].index

o_kontrolu_on_ucuz
o_kontrolu_on_ucuz = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Ucuz") & (df["3G"] == "Var")].index

o_kontrolu_on_ucuz
df.loc[o_kontrolu_on_ucuz,"OnKameraMP"] = 4.95
o_kontrolu_on_ucuz = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Ucuz")& (df["3G"] == "Var")].index

o_kontrolu_on_ucuz
o_kontrolu_on_ucuz = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Ucuz")]

o_kontrolu_on_ucuz
o_kontrolu_on_ucuz = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Ucuz")& (df["ArkaKameraMP"] > 10)].index

o_kontrolu_on_ucuz
df.loc[o_kontrolu_on_ucuz,"OnKameraMP"] = 4.95
o_kontrolu_on_ucuz = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Ucuz")]

o_kontrolu_on_ucuz
o_kontrolu_on_cok_ucuz = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Çok Ucuz")]

o_kontrolu_on_cok_ucuz
o_kontrolu_on_cok_ucuz = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Çok Ucuz")& (df["4G"] == "Var")].index

o_kontrolu_on_cok_ucuz
df.groupby("FiyatAraligi").mean()
df.loc[o_kontrolu_on_cok_ucuz,"OnKameraMP"] = 4
o_kontrolu_on_cok_ucuz = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Çok Ucuz")& (df["3G"] == "Var")].index

o_kontrolu_on_cok_ucuz
df.loc[o_kontrolu_on_cok_ucuz,"OnKameraMP"] = 4
o_kontrolu_on_cok_ucuz = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Çok Ucuz")]

o_kontrolu_on_cok_ucuz
df.groupby("FiyatAraligi").mean()
o_kontrolu_on_cok_ucuz = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Çok Ucuz")& (df["ArkaKameraMP"] > 9)].index

o_kontrolu_on_cok_ucuz
df.loc[o_kontrolu_on_cok_ucuz,"OnKameraMP"] = 4
o_kontrolu_on_cok_ucuz = df[(df["OnKameraMP"] == 0) & (df["FiyatAraligi"] == "Çok Ucuz")]

o_kontrolu_on_cok_ucuz
o_kontrolu_on_ucuz = df[(df["DahiliBellek"] == 0)]

o_kontrolu_on_ucuz
o_kontrolu = df[(df["Kalinlik"] == 0)]

o_kontrolu
o_kontrolu = df[(df["Agirlik"] == 0)]

o_kontrolu
o_kontrolu = df[(df["CekirdekSayisi"] == 0)]

o_kontrolu
o_kontrolu = df[(df["ArkaKameraMP"] == 0)]

o_kontrolu
o_kontrolu_arka_pahali = df[(df["ArkaKameraMP"] == 0) & (df["FiyatAraligi"] == "Pahalı")].index

o_kontrolu_arka_pahali
df.groupby("FiyatAraligi").mean()
df.loc[o_kontrolu_arka_pahali,"ArkaKameraMP"] = 10.1
o_kontrolu_arka_normal = df[(df["ArkaKameraMP"] == 0) & (df["FiyatAraligi"] == "Normal")].index

o_kontrolu_arka_normal
df.loc[o_kontrolu_arka_normal,"ArkaKameraMP"] = 10
o_kontrolu_arka_ucuz = df[(df["ArkaKameraMP"] == 0) & (df["FiyatAraligi"] == "Ucuz")]

o_kontrolu_arka_ucuz
df.groupby("FiyatAraligi").mean()
o_kontrolu_arka_ucuz = df[(df["ArkaKameraMP"] == 0) & (df["FiyatAraligi"] == "Ucuz")].index

o_kontrolu_arka_ucuz
df.loc[o_kontrolu_arka_ucuz,"ArkaKameraMP"] = 9.9
o_kontrolu_arka_ucuz = df[(df["ArkaKameraMP"] == 0) & (df["FiyatAraligi"] == "Ucuz")& (df["WiFi"] == "Var")].index

o_kontrolu_arka_ucuz
o_kontrolu_arka_cok_ucuz = df[(df["ArkaKameraMP"] == 0) & (df["FiyatAraligi"] == "Çok Ucuz")]

o_kontrolu_arka_cok_ucuz
o_kontrolu_arka_cok_ucuz = df[(df["ArkaKameraMP"] == 0) & (df["FiyatAraligi"] == "Çok Ucuz")& (df["4G"] == "Var")].index

o_kontrolu_arka_cok_ucuz
df.groupby("FiyatAraligi").mean()
df.loc[o_kontrolu_arka_cok_ucuz,"ArkaKameraMP"] = 9.5
o_kontrolu_arka_cok_ucuz = df[(df["ArkaKameraMP"] == 0) & (df["FiyatAraligi"] == "Çok Ucuz")& (df["3G"] == "Var")].index

o_kontrolu_arka_cok_ucuz
df.loc[o_kontrolu_arka_cok_ucuz,"ArkaKameraMP"] = 9.5
o_kontrolu_arka_cok_ucuz = df[(df["ArkaKameraMP"] == 0) & (df["FiyatAraligi"] == "Çok Ucuz")]

o_kontrolu_arka_cok_ucuz
o_kontrolu_arka_cok_ucuz = df[(df["ArkaKameraMP"] == 0) & (df["FiyatAraligi"] == "Çok Ucuz")].index

o_kontrolu_arka_cok_ucuz
df.loc[o_kontrolu_arka_cok_ucuz,"ArkaKameraMP"] = 5
o_kontrolu = df[(df["CozunurlukYükseklik"] == 0)]

o_kontrolu
df.groupby("FiyatAraligi").mean()
df.loc[662,"CozunurlukYükseklik"] = 666
df.loc[856,"CozunurlukYükseklik"] = 744
o_kontrolu = df[(df["CozunurlukYükseklik"] == 0)]

o_kontrolu
o_kontrolu = df[(df["CozunurlukGenislik"] == 0)]

o_kontrolu
o_kontrolu = df[(df["BataryaOmru"] == 0)]

o_kontrolu
df.head()
label_encoder = preprocessing.LabelEncoder()
df['Bluetooth'] = label_encoder.fit_transform(df['Bluetooth'])

df.head()
blue_1_olanlar = df[(df["Bluetooth"] == 1)].index

blue_0_olanlar = df[(df["Bluetooth"] == 0)].index

df.loc[blue_1_olanlar,"Bluetooth"] = 0

df.loc[blue_0_olanlar,"Bluetooth"] = 1
df.head()
df['4G'] = label_encoder.fit_transform(df['4G'])

df.head()
dortg_1_olanlar = df[(df["4G"] == 1)].index

dortg_0_olanlar = df[(df["4G"] == 0)].index

df.loc[dortg_1_olanlar,"4G"] = 0

df.loc[dortg_0_olanlar,"4G"] = 1
df.head()
df['WiFi'] = label_encoder.fit_transform(df['WiFi'])

df.head()
wifi_1_olanlar = df[(df["WiFi"] == 1)].index

wifi_0_olanlar = df[(df["WiFi"] == 0)].index

df.loc[wifi_1_olanlar,"WiFi"] = 0

df.loc[wifi_0_olanlar,"WiFi"] = 1

df.head()
df['3G'] = label_encoder.fit_transform(df['3G'])

df.head()
ucg_1_olanlar = df[(df["3G"] == 1)].index

ucg_0_olanlar = df[(df["3G"] == 0)].index

df.loc[ucg_1_olanlar,"3G"] = 0

df.loc[ucg_0_olanlar,"3G"] = 1

df.head()
df['Dokunmatik'] = label_encoder.fit_transform(df['Dokunmatik'])

df.head()
dokunmatik_1_olanlar = df[(df["Dokunmatik"] == 1)].index

dokunmatik_0_olanlar = df[(df["Dokunmatik"] == 0)].index

df.loc[dokunmatik_1_olanlar,"Dokunmatik"] = 0

df.loc[dokunmatik_0_olanlar,"Dokunmatik"] = 1

df.head()
df['CiftHat'] = label_encoder.fit_transform(df['CiftHat'])

df.head()
cifthat_1_olanlar = df[(df["CiftHat"] == 1)].index

cifthat_0_olanlar = df[(df["CiftHat"] == 0)].index

df.loc[cifthat_1_olanlar,"CiftHat"] = 0

df.loc[cifthat_0_olanlar,"CiftHat"] = 1

df.head()
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, precision_score

from sklearn.naive_bayes import GaussianNB
df.drop("Renk", axis = 1, inplace = True)## Renk ile fiyatın alakalı olma durumunu elediğimiz için renk öznieliğini 

#veri setinden çıkartıyoruz.

y = df['FiyatAraligi']

X = df.drop(['FiyatAraligi'], axis=1)

df.head()
y.head()
X.head()
X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y, 

                                                    test_size = 0.25, 

                                                    random_state = 42)
X_train.info() # x train veri setimiz 1500 gözlemden oluşuyor.
X_test.info() # x test veri setimiz ise 500 gözlem içeriyor. %25 olduğunu buradan da görüyoruz.
y_train.value_counts() # Bagımlı değişken olan y değerimiz sürekli olmadığı için info fonksiyonunu kullanmayıp 

#value_counts kullanıyoruz
y_test.value_counts() # y train ile y testi karşılaştırdığımızda yine %25 olduğunu görüyoruz. 
df.head()
nb = GaussianNB()

nb_model = nb.fit(X_train, y_train)
fiyat_tahmin_nb = nb_model.predict(X_test)
accuracy_score(y_test, fiyat_tahmin_nb)
karmasiklik_matrisi = confusion_matrix(y_test, fiyat_tahmin_nb)

print(karmasiklik_matrisi)
cross_val_score(nb_model, X_test, y_test, cv = 10)
cross_val_score(nb_model,X_test, y_test, cv = 10).mean()
PrecisionScore = precision_score(y_test, fiyat_tahmin_nb, average='weighted')

PrecisionScore
RecallScore = recall_score(y_test, fiyat_tahmin_nb, average='weighted')

RecallScore
F1Score = f1_score(y_test, fiyat_tahmin_nb, average = 'weighted')  

F1Score
print(classification_report(y_test, fiyat_tahmin_nb))
from numpy import set_printoptions

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif



test = SelectKBest(score_func = f_classif, k = 'all')
fit = test.fit(X, y)
fit # k parametremizin all olduğunu görüyoruz.
set_printoptions(precision = 3)

print(fit.scores_.astype(int))
X.columns
y = df['FiyatAraligi']

X = pd.concat([df["RAM"],df["BataryaGucu"], df["CozunurlukGenislik"],df["CozunurlukYükseklik"],df["Agirlik"]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y, 

                                                    test_size = 0.25, 

                                                    random_state = 42)
nb = GaussianNB()

nb_model = nb.fit(X_train, y_train)
nb_model # priors özelliğinin none olduğunu görüyoruz.
fiyat_tahmini_nb_sifirsiz= nb_model.predict(X_test)
accuracy_score(y_test, fiyat_tahmini_nb_sifirsiz)
accuracy_score(y_test, fiyat_tahmin_nb)
karmasiklik_matrisi = confusion_matrix(y_test, fiyat_tahmini_nb_sifirsiz)

print(karmasiklik_matrisi)
cross_val_score(nb_model, X_test, y_test, cv = 10) # 10 veri atlayarak test ettik.
cross_val_score(nb_model, X_test, y_test, cv = 10).mean() ## verilerimizin ortalamlarını alalım. Ve sonucumuzu skorumuz ile karşılaştıralım.

PrecisionScore = precision_score(y_test, fiyat_tahmini_nb_sifirsiz, average='weighted')

PrecisionScore
RecallScore = recall_score(y_test, fiyat_tahmini_nb_sifirsiz, average='weighted')

RecallScore
F1Score = f1_score(y_test, fiyat_tahmini_nb_sifirsiz, average = 'weighted')  

F1Score
print(classification_report(y_test,fiyat_tahmini_nb_sifirsiz)) 
print(classification_report(y_test, fiyat_tahmin_nb)) # Bir önceki raporlama
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import recall_score, f1_score, precision_score

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

from sklearn import ensemble

from sklearn.metrics import confusion_matrix as cm

from matplotlib.legend_handler import HandlerLine2D
df.head()
y = df["FiyatAraligi"]

X = df.drop(["FiyatAraligi"], axis=1)
y.head()
y.value_counts() # dagılımın dengeli olduğunu görüyoruz.
sns.countplot(x = "FiyatAraligi", data = df) 
X.head()
X.sample()
X_train.sample() # kontrol amaçlı kullandım.
y_test.sample() # kontrol amaçlı kullandım.
X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.25, 

                                                    random_state=42)
cart_grid = {"max_depth": range(1,20),

            "min_samples_split" : range(2,50)}
cart = DecisionTreeClassifier(random_state = 42 ,criterion="gini")

cart_cv = GridSearchCV(cart, cart_grid, cv = 10, n_jobs = -1, verbose = 2)
cart_cv_model = cart_cv.fit(X_train, y_train)# Parametre iyileştirmesi yapabilmek için ilk eğitimimiz.
print("En iyi parametreler : " + str(cart_cv_model.best_params_))

print("En iyi skor : " + str(cart_cv_model.best_score_))
cart = DecisionTreeClassifier(max_depth = 8, min_samples_split = 2)

cart_tuned = cart.fit(X_train, y_train)
X.columns
from sklearn.tree.export import export_text

r = export_text(cart, feature_names = ['BataryaGucu', 'Bluetooth', 'MikroislemciHizi', 'CiftHat', 'OnKameraMP',

       '4G', 'DahiliBellek', 'Kalinlik', 'Agirlik', 'CekirdekSayisi',

       'ArkaKameraMP', 'CozunurlukYükseklik', 'CozunurlukGenislik', 'RAM',

       'BataryaOmru', '3G', 'Dokunmatik', 'WiFi'])

print(r)

import os

os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\bin'

os.environ["PATH"]
from sklearn.tree import export_graphviz

from sklearn import tree

from IPython.display import SVG

from graphviz import Source

from IPython.display import display



graph = Source(tree.export_graphviz(cart, out_file = None, feature_names = X.columns, filled = True))

display(SVG(graph.pipe(format = 'svg')))
y_pred = cart_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
cross_val_score(cart_tuned, X_test, y_test, cv = 10)
cross_val_score(cart_tuned, X, y, cv = 10).mean()
print(classification_report(y_test, y_pred))
ranking = cart.feature_importances_

features = np.argsort(ranking)[::-1][:10]

columns = X.columns



plt.figure(figsize = (16, 9))

plt.title("Karar Ağacına Göre Özniteliklerin Önem Derecesi", y = 1.03, size = 18)

plt.bar(range(len(features)), ranking[features], color="lime", align="center")

plt.xticks(range(len(features)), columns[features], rotation=80)

plt.show()
print(classification_report(y_test, y_pred)) # Gini parametremizin verdiği sonuçlar
X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.25, 

                                                    random_state=42)
cart_grid = {"max_depth": range(1,20),

            "min_samples_split" : range(2,50)}
cart = DecisionTreeClassifier(random_state = 42, criterion="entropy")

cart_cv = GridSearchCV(cart, cart_grid, cv = 10, n_jobs = -1, verbose = 2)
cart_cv_model = cart_cv.fit(X_train, y_train)
cart = DecisionTreeClassifier(max_depth = 8, min_samples_split = 2) # Gini ile aldığımız en iyi parametreler bilgisinde dolduruldu.

cart_tuned = cart.fit(X_train, y_train)
y_pred_ent = cart_tuned.predict(X_test)
accuracy_score(y_test, y_pred_ent) ## entropy skoru
accuracy_score(y_test, y_pred) ## gini skoru
karmasiklik_matrisi = confusion_matrix(y_test, y_pred_ent)

print(karmasiklik_matrisi)
print(classification_report(y_test, y_pred_ent)) # entropy parametremizin verdigi sonuclar
print(classification_report(y_test, y_pred)) # Gini parametremizin verdiği sonuçlar
knn_params = {"n_neighbors": np.arange(1,15)} # Komşularda 1'den başlayıp 14'e kadar hepsini deneyecek.
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, knn_params, cv = 3)

knn_cv.fit(X_train, y_train)
print("En iyi skor: " + str(knn_cv.best_score_))

print("En iyi parametreler: " + str(knn_cv.best_params_))
knn = KNeighborsClassifier(9)

knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
score_list = []



for each in range(1,30):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(X_train,y_train)

    score_list.append(knn2.score(X_test, y_test))

    

plt.plot(range(1,30),score_list)

plt.xlabel("k değerleri")

plt.ylabel("doğruluk skoru")

plt.show()
komsu_indeksleri = []

skorlar = []

temp = 2

while temp <= 15:

    knn = KNeighborsClassifier(temp)

    knn_tuned = knn.fit(X_train, y_train)

    y_pred = knn_tuned.predict(X_test)

    komsu_indeksleri.append(temp)

    skorlar.append(accuracy_score(y_test, y_pred))

    temp +=1

        

komsular = pd.DataFrame({'Komşu İndeksi':komsu_indeksleri, 

                    'Skor':skorlar})     
sns.lmplot(x = "Komşu İndeksi", y = "Skor", data = komsular)
sns.scatterplot(x = "Komşu İndeksi", y = "Skor", data = komsular)
knn = KNeighborsClassifier(11)

knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred_ent))