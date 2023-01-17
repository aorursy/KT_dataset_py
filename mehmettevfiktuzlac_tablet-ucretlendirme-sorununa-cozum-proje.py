import numpy as np

import seaborn as sns

import pandas as pd

import missingno  

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, precision_score

from sklearn.naive_bayes import GaussianNB



from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from warnings import filterwarnings

from sklearn import ensemble

from sklearn.metrics import confusion_matrix as cm



filterwarnings('ignore')



from matplotlib.legend_handler import HandlerLine2D
df = pd.read_csv("../input/tabletpc-priceclassification/tablet.csv")
df.head() #ilk beş gözlemi getirerek inceleyelim
df.shape #2000 gözlem ve 20 adet öznitelikten oluşan bir datamız var
df.isna().sum()
df.info()
df.describe().T
df["OnKameraMP"].unique()
df[(df["OnKameraMP"] == 0)]
df["ArkaKameraMP"].unique()
df[(df["ArkaKameraMP"] == 0)]
df["CozunurlukYükseklik"].nunique() #1137 adet benzersiz değer var bunları yazdırıpta incelemek çok zor olacaktır.
df[(df["CozunurlukYükseklik"] == 0)]
fig = plt.gcf()

fig.set_size_inches(11, 7)

sns.scatterplot(x = "CozunurlukYükseklik", y = "CozunurlukGenislik",hue='FiyatAraligi', data = df);
df.corr() 
corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
df.corr()["ArkaKameraMP"]["OnKameraMP"]
df['FiyatAraligi'].value_counts()
sns.countplot(x = "FiyatAraligi", data = df);
sns.distplot(df["ArkaKameraMP"], bins=16, color="purple");
sns.distplot(df["MikroislemciHizi"], bins=16, color="gold");
sns.distplot(df["OnKameraMP"], bins=16, color="blue");
sns.violinplot(x = "FiyatAraligi", y = "RAM", data = df);
sns.violinplot(x = "FiyatAraligi", y = "BataryaGucu", data = df); 
sns.violinplot(x = "FiyatAraligi", y = "DahiliBellek", data = df); 
sns.violinplot(x = "FiyatAraligi", y = "MikroislemciHizi", data = df);
df.axes # hangi öznitelikler vardı
sns.countplot(df["Bluetooth"]);
sns.countplot(df["Dokunmatik"]);
sns.countplot(df["4G"]);
sns.factorplot("FiyatAraligi", "CozunurlukYükseklik","WiFi", data = df, kind = "bar");
df.columns
sns.countplot(df["FiyatAraligi"]); #önceden bunu incelemiştik zaten dengeli bir şekilde olması bizim avantaj
df.isna().sum()
missingno.matrix(df,figsize=(20, 10));
missingno.heatmap(df, figsize= (15,8));
def eksik_deger_tablosu(df): 

    eksik_deger = df.isnull().sum()

    eksik_deger_yuzde = 100 * df.isnull().sum()/len(df)

    eksik_deger_tablo = pd.concat([eksik_deger, eksik_deger_yuzde], axis=1)

    eksik_deger_tablo_son = eksik_deger_tablo.rename(columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})

    return eksik_deger_tablo_son
eksik_deger_tablosu(df)
df[(df["BataryaOmru"] < 0)]
df["WiFi"].unique()
df[(df["CozunurlukYükseklik"] == 0)]
sns.countplot(df[df["OnKameraMP"].isnull()]["FiyatAraligi"]);
df.groupby("FiyatAraligi")[["OnKameraMP"]].mean()
sns.countplot(df[df["RAM"].isnull()]["FiyatAraligi"]);
df.groupby("FiyatAraligi")[["RAM"]].mean()
df[(df["FiyatAraligi"] == "Çok Ucuz") & (df["OnKameraMP"].isnull())]
df[(df["FiyatAraligi"] == "Pahalı") & (df["RAM"].isnull())]
CokUcuz_OnKameraMP = df[(df["FiyatAraligi"] == "Çok Ucuz") & (df["OnKameraMP"].isnull())].index
Pahalı_RAM = df[(df["FiyatAraligi"] == "Pahalı") & (df["RAM"].isnull())].index
df.loc[CokUcuz_OnKameraMP ,"OnKameraMP"] = 4.1

df.loc[Pahalı_RAM, "RAM"] = 3449

#burada çok küçük bir yuvarlama işlemi yaptım 4.09 -> 4.1 ve RAM miktarındaki kesirli kısım bunlar diğer türlü koysaydım da arada fazla bir fark olmazdı
df.isna().sum()
oran = df['CozunurlukYükseklik'] / df['CozunurlukGenislik']
df['ekranOrani'] = oran

df
df[(df["ekranOrani"] < 0.25)]
df['CozunurlukYükseklik'].std()
df["ekranOrani"].mean()
df.groupby("FiyatAraligi")[["ekranOrani"]].mean()
df.groupby("FiyatAraligi")[["CozunurlukYükseklik"]].mean()
df.groupby("FiyatAraligi")[["CozunurlukGenislik"]].mean()
pahali_cozunurluk = df[(df["FiyatAraligi"] == "Pahalı") & (df["ekranOrani"] < 0.25)].index

ucuz_cozunurluk = df[(df["FiyatAraligi"] == "Ucuz") & (df["ekranOrani"] < 0.25)].index

normal_cozunurluk = df[(df["FiyatAraligi"] == "Normal") & (df["ekranOrani"] < 0.25)].index

cokUcuz_cozunurluk = df[(df["FiyatAraligi"] == "Çok Ucuz") & (df["ekranOrani"] < 0.25)].index
df.loc[pahali_cozunurluk ,"CozunurlukYükseklik"] = df['CozunurlukGenislik'] * 0.510424

df.loc[ucuz_cozunurluk ,"CozunurlukYükseklik"] = df['CozunurlukGenislik'] * 0.534434

df.loc[normal_cozunurluk ,"CozunurlukYükseklik"] = df['CozunurlukGenislik'] * 0.510424

df.loc[cokUcuz_cozunurluk ,"CozunurlukYükseklik"] = df['CozunurlukGenislik'] * 0.479679
df.head(10)
df["CozunurlukYükseklik"] = df["CozunurlukYükseklik"].astype(int)
oran2 = df['CozunurlukYükseklik'] / df['CozunurlukGenislik']

df['ekranOrani'] = oran2

df.sample(10)
df["CozunurlukYükseklik"].std()
df["ekranOrani"].mean()
df.groupby("FiyatAraligi")[["ekranOrani"]].mean()
df.groupby("FiyatAraligi")[["CozunurlukYükseklik"]].mean()
df.drop(["ekranOrani"], axis = 1, inplace = True)
df.sample(10)
sns.boxplot(x = df['OnKameraMP']);
sns.boxplot(x = df['CozunurlukYükseklik']);
df.columns
df['Bluetooth'] = pd.Categorical(df['Bluetooth'])

dfDummies = pd.get_dummies(df['Bluetooth'], prefix = 'Bluetooth')

dfDummies
df = pd.concat([df, dfDummies], axis=1)

df.head()
df.drop(['Bluetooth', 'Bluetooth_Yok'], axis = 1, inplace = True)
df['CiftHat'] = pd.Categorical(df['CiftHat'])

dfDummies = pd.get_dummies(df['CiftHat'], prefix = 'CiftHat')

df = pd.concat([df, dfDummies], axis=1)

df.drop(['CiftHat', 'CiftHat_Yok'], axis = 1, inplace = True)
df['4G'] = pd.Categorical(df['4G'])

dfDummies = pd.get_dummies(df['4G'], prefix = '4G')

df = pd.concat([df, dfDummies], axis=1)

df.drop(['4G', '4G_Yok'], axis = 1, inplace = True)
df['3G'] = pd.Categorical(df['3G'])

dfDummies = pd.get_dummies(df['3G'], prefix = '3G')

df = pd.concat([df, dfDummies], axis=1)

df.drop(['3G', '3G_Yok'], axis = 1, inplace = True) 
df['Dokunmatik'] = pd.Categorical(df['Dokunmatik'])

dfDummies = pd.get_dummies(df['Dokunmatik'], prefix = 'Dokunmatik')

df = pd.concat([df, dfDummies], axis=1)

df.drop(['Dokunmatik', 'Dokunmatik_Yok'], axis = 1, inplace = True)
df['WiFi'] = pd.Categorical(df['WiFi'])

dfDummies = pd.get_dummies(df['WiFi'], prefix = 'WiFi')

df = pd.concat([df, dfDummies], axis=1)

df.drop(['WiFi', 'WiFi_Yok'], axis = 1, inplace = True)
df
corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
df['Renk'] = pd.Categorical(df['Renk'])

dfDummies = pd.get_dummies(df['Renk'], prefix = 'Renk')

df = pd.concat([df, dfDummies], axis=1)
df.drop(['Renk', 'Renk_Yeşil'], axis = 1, inplace = True)
df.columns
y = df['FiyatAraligi'] # hedef özniteliğimiz.

X = df.drop(['FiyatAraligi'], axis=1) # çıkarımda bulunacağımız özniteliklerimiz.
y
X
X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y, 

                                                    test_size = 0.25, 

                                                    random_state = 42)
X_train.head()
X_test.head()
y_train.head()
y_test.head()
nb = GaussianNB()

nb_model = nb.fit(X_train, y_train)
nb_model 
y_pred = nb_model.predict(X_test)
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
from numpy import set_printoptions

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif
len(X.columns)
test = SelectKBest(k = 29)

test
fit = test.fit(X, y)

fit
for indis, skor in enumerate(fit.scores_):

    print(skor, " -> ", X.columns[indis])
y = df['FiyatAraligi']

X = df[["BataryaGucu", "CozunurlukYükseklik", "CozunurlukGenislik", "RAM", "WiFi_Var", "Dokunmatik_Var", "4G_Var", "3G_Var", "CiftHat_Var", "Bluetooth_Var", "BataryaOmru", "ArkaKameraMP", "CekirdekSayisi", "Agirlik", "Kalinlik", "DahiliBellek", "OnKameraMP", "MikroislemciHizi"]]
X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y, 

                                                    test_size = 0.25, 

                                                    random_state = 42)
nb = GaussianNB()

nb_model = nb.fit(X_train, y_train)

nb_model
y_pred = nb_model.predict(X_test)
accuracy_score(y_test, y_pred)
cross_val_score(nb_model, X_test, y_test, cv = 10)
cross_val_score(nb_model, X_test, y_test, cv = 10).mean()
PrecisionScore = precision_score(y_test, y_pred, average='weighted')

PrecisionScore
RecallScore = recall_score(y_test, y_pred, average='weighted')

RecallScore
F1Score = f1_score(y_test, y_pred, average = 'weighted')  

F1Score
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
cart = DecisionTreeClassifier(random_state = 42)

cart_model = cart.fit(X_train, y_train)
cart_model
!pip install skompiler

!pip install graphviz

!pip install pydotplus

!pip install astor



from skompiler import skompile

print(skompile(cart_model.predict).to("python/code"))
y_pred = cart_model.predict(X_test)
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
cross_val_score(cart_model, X_test, y_test, cv = 10)
cross_val_score(cart_model, X_test, y_test, cv = 10).mean()
print(classification_report(y_test, y_pred))
cart = DecisionTreeClassifier(random_state = 42, criterion='entropy')

cart_model = cart.fit(X_train, y_train)
cart_model #criterion='entropy' olduğunu görüyoruz.
y_pred = cart_model.predict(X_test)
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
cross_val_score(cart_model, X_test, y_test, cv = 10)
cross_val_score(cart_model, X_test, y_test, cv = 10).mean()
print(classification_report(y_test, y_pred))
import os

os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38'

os.environ["PATH"]
from sklearn.tree import export_graphviz

from sklearn import tree

from IPython.display import SVG

from graphviz import Source

from IPython.display import display

graph = Source(tree.export_graphviz(cart, out_file = None, feature_names = X.columns, filled = True, rounded = True))

display(SVG(graph.pipe(format = 'svg')))
ranking = cart.feature_importances_

features = np.argsort(ranking)[::-1][:10]

columns = X.columns



plt.figure(figsize = (16, 9))

plt.title("Karar Ağacına Göre Özniteliklerin Önem Derecesi", y = 1.03, size = 18)

plt.bar(range(len(features)), ranking[features], color="lime", align="center")

plt.xticks(range(len(features)), columns[features], rotation=80)

plt.show()
knn = KNeighborsClassifier()

knn_model = knn.fit(X_train, y_train)

knn_model
y_pred = knn_model.predict(X_test)
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
cross_val_score(knn_model, X_test, y_test, cv = 10)
cross_val_score(knn_model, X_test, y_test, cv = 10).mean()
knn = range(2,15)

for k_degeri in knn:

    knn = KNeighborsClassifier(n_neighbors = k_degeri)

    knn_model = knn.fit(X_train, y_train)

    skor = knn_model.score(X_test, y_test)

    print("k=", k_degeri ,"degeri için ->", skor)
score_list = []



for each in range(2,15,1):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(X_train,y_train)

    score_list.append(knn2.score(X_test, y_test))



plt.plot(range(2,15,1),score_list)

plt.xlabel("k en yakın komşu sayıları")

plt.ylabel("doğruluk skoru")

plt.show()
knn = KNeighborsClassifier(n_neighbors = 13)

knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
cross_val_score(knn_model, X_test, y_test, cv = 10)
cross_val_score(knn_model, X_test, y_test, cv = 10).mean()
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)