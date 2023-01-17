

#KEŞİFÇİ VERİ ANALİZİ



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



#VERİ ÖN İŞLEME



from collections import Counter

import missingno                   

from sklearn import preprocessing    

import re                        

import matplotlib.pyplot as plt



#MODEL EĞİTİMİ

from sklearn.linear_model import LinearRegression

linear_regresyon=LinearRegression()

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 

import math

from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures(degree = 2)

#Gaussnb

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, precision_score

from sklearn.naive_bayes import GaussianNB

#Decision Tree

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report

from sklearn.metrics import roc_auc_score,roc_curve

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import recall_score, f1_score, precision_score

from sklearn.tree import DecisionTreeClassifier

from warnings import filterwarnings

import matplotlib.pyplot as plt

from sklearn import ensemble

from sklearn.metrics import confusion_matrix as cm

from matplotlib.legend_handler import HandlerLine2D



from sklearn.tree import export_graphviz

from sklearn import tree

from IPython.display import SVG

from graphviz import Source

from IPython.display import display



#KNN

#filterwarnings('ignore')

from warnings import filterwarnings

import matplotlib.pyplot as plt

from sklearn import ensemble

from sklearn.metrics import confusion_matrix as cm

filterwarnings('ignore')
tablet= pd.read_csv("../input/tabletkategoriletirme/tablet.csv",sep = ",")

df=tablet.copy()
df.head()
df.sample(10)
df.shape
df.info()
df.isna().sum()
df["BataryaGucu"].unique()
df["Bluetooth"].unique()
df["MikroislemciHizi"].unique()
df["CiftHat"].unique()
df["OnKameraMP"].unique()
df["4G"].unique()
df["DahiliBellek"].unique()
df["Kalinlik"].unique()
df["Agirlik"].unique()
df["CekirdekSayisi"].unique()
df["ArkaKameraMP"].unique()
df["CozunurlukYükseklik"].unique()
df["CozunurlukGenislik"].unique()
df["RAM"].unique()
df["BataryaOmru"].unique()
df["CekirdekSayisi"].unique()
df["3G"].unique()
df["WiFi"].unique()
df["FiyatAraligi"].unique()
df["Renk"].unique()
df.count()
df.describe().T
df.columns
sns.countplot(df["Bluetooth"]);
sns.countplot(df["CiftHat"]);
sns.countplot(df["4G"]);
sns.countplot(df["3G"]);
sns.countplot(df["Dokunmatik"]);
sns.countplot(df["WiFi"]);
sns.countplot(df["FiyatAraligi"]);
sns.distplot(df["BataryaGucu"], bins=12, color="purple");
sns.distplot(df["MikroislemciHizi"], bins=12, color="purple");
sns.distplot(df["OnKameraMP"], bins=12, color="purple");
sns.distplot(df["DahiliBellek"], bins=12, color="purple");
sns.distplot(df["Kalinlik"], bins=12, color="purple");
sns.distplot(df["Agirlik"], bins=12, color="purple");
sns.distplot(df["CekirdekSayisi"], bins=12, color="purple");
sns.distplot(df["ArkaKameraMP"], bins=12, color="purple");
sns.distplot(df["CozunurlukYükseklik"], bins=12, color="purple");
sns.violinplot(df["CozunurlukGenislik"], bins=12, color="purple");
sns.distplot(df["BataryaOmru"], bins=12, color="purple");
sns.distplot(df["RAM"], bins=12, color="purple");
df.corr()
df.cov()
corr=df.corr()

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
df.corr()["OnKameraMP"]["ArkaKameraMP"]
sns.scatterplot(x="ArkaKameraMP",y="OnKameraMP",data=df);
sns.jointplot(x="ArkaKameraMP",y="OnKameraMP",data=df);
sns.scatterplot(x="ArkaKameraMP",y="OnKameraMP",data=df,hue="FiyatAraligi");
sns.violinplot(y = "ArkaKameraMP", data = df);
sns.violinplot(y = "OnKameraMP", data = df);
sns.violinplot(x = "FiyatAraligi", y = "ArkaKameraMP", data = df);
sns.violinplot(x = "FiyatAraligi", y = "OnKameraMP", data = df);
sns.jointplot(x = df["ArkaKameraMP"], y = df["OnKameraMP"], color = "darkblue");
sns.jointplot(x = df["ArkaKameraMP"], y = df["OnKameraMP"], kind = "kde", color = "red");
sns.distplot(df["ArkaKameraMP"], bins=12 ,color="darkblue");
sns.distplot(df["OnKameraMP"], bins=12 ,color="darkblue");
sns.lmplot(y = "OnKameraMP", x = "ArkaKameraMP", data = df);
df["FiyatAraligi"].value_counts() #burda hepsinin dengelimi dağıldığına sayısal olarak bakacağız
sns.countplot(x="FiyatAraligi",data=df)
df.groupby(["FiyatAraligi"]).mean()
df.groupby(["FiyatAraligi"]).mean()["RAM"]
df.groupby(["FiyatAraligi"]).mean()["OnKameraMP"]
df.groupby(["FiyatAraligi"]).std()["OnKameraMP"]
df.groupby(["FiyatAraligi"]).mean()["ArkaKameraMP"]
df.groupby(["FiyatAraligi"]).std()["ArkaKameraMP"]
df.isnull().sum()
df.isnull().sum().sum()
missingno.matrix(df,figsize=(20, 10));
missingno.heatmap(df, figsize= (15,8));
sns.countplot(df[df["OnKameraMP"].isnull()]["RAM"]);
def eksik_deger_tablosu(df): 

    eksik_deger = df.isnull().sum()

    eksik_deger_yuzde = 100 * df.isnull().sum()/len(df)

    eksik_deger_tablo = pd.concat([eksik_deger, eksik_deger_yuzde], axis=1)

    eksik_deger_tablo_son = eksik_deger_tablo.rename(columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})

    return eksik_deger_tablo_son
eksik_deger_tablosu(df)
sns.countplot(df[df["OnKameraMP"].isnull()]["FiyatAraligi"]);
sns.countplot(df[df["RAM"].isnull()]["FiyatAraligi"]);
df.groupby("FiyatAraligi").mean()
df.groupby("FiyatAraligi")[["OnKameraMP"]].mean()
df[(df["FiyatAraligi"] == "Çok Ucuz") & (df["OnKameraMP"].isnull())]
sns.violinplot(y = "OnKameraMP", data = df);
sns.distplot(df["OnKameraMP"], bins=12 ,color="darkblue");
eksik_onkameramp_indexleri=df[(df["FiyatAraligi"] == "Çok Ucuz") & (df["OnKameraMP"].isnull())].index

eksik_onkameramp_indexleri
df.loc[eksik_onkameramp_indexleri ,"OnKameraMP"] = 4
eksik_deger_tablosu(df)
df["OnKameraMP"].unique()
sns.violinplot(y = "OnKameraMP", data = df);
sns.distplot(df["OnKameraMP"], bins=12 ,color="darkblue");
df.groupby("FiyatAraligi").mean()
df.groupby("FiyatAraligi")[["RAM"]].mean()
df[(df["FiyatAraligi"] == "Pahalı") & (df["RAM"].isnull())]
sns.violinplot(y = "RAM", data = df);
sns.distplot(df["RAM"], bins=12 ,color="darkblue");
eksik_RAM_indexleri=df[(df["FiyatAraligi"] == "Pahalı") & (df["RAM"].isnull())].index

eksik_RAM_indexleri
df.loc[eksik_RAM_indexleri ,"RAM"] = 3450
eksik_deger_tablosu(df)
df["RAM"].unique()
sns.violinplot(y = "RAM", data = df);
sns.distplot(df["RAM"], bins=12 ,color="darkblue");
df.isna().sum()
df["RAM"].unique()
df["OnKameraMP"].unique()
sns.boxplot(x = df['BataryaGucu']);
sns.boxplot(x = df['MikroislemciHizi']);
sns.boxplot(x = df['DahiliBellek']);
sns.boxplot(x = df['OnKameraMP']);
sns.boxplot(x = df['Kalinlik']);
sns.boxplot(x = df['Agirlik']);
sns.boxplot(x = df['CekirdekSayisi']);
sns.boxplot(x = df['ArkaKameraMP']);
sns.boxplot(x = df['CozunurlukGenislik']);
sns.boxplot(x = df['CozunurlukYükseklik']);
sns.boxplot(x = df['RAM']);
sns.boxplot(x = df['BataryaOmru']);
Q1 = df.OnKameraMP.quantile(0.25) 

Q2 = df.OnKameraMP.quantile(0.5) 

Q3 = df.OnKameraMP.quantile(0.75)

Q4 = df.OnKameraMP.quantile(1)



IQR = Q3 - Q1
alt_sinir = Q1 - 1.5 * IQR

ust_sinir = Q3 + 1.5 * IQR
print("Q1 ->", Q1)

print("Q3 ->", Q3)

print("Q2 ->", Q2)

print("Q4 ->", Q4)

print("IQR ->", IQR)



print()



print("Alt sınır: Q1 - 1.5 * IQR ->", alt_sinir)

print("Üst sınır: Q3 + 1.5 * IQR ->", ust_sinir)
outliers_df = df[(df["OnKameraMP"] < alt_sinir) | (df["OnKameraMP"] > ust_sinir)]

outliers_df
outliers_df.index
df.drop(outliers_df.index,inplace=True)
sns.boxplot(x = df['OnKameraMP']);
Q1 = df.CozunurlukYükseklik.quantile(0.25) 

Q2 = df.CozunurlukYükseklik.quantile(0.5) 

Q3 = df.CozunurlukYükseklik.quantile(0.75)

Q4 = df.CozunurlukYükseklik.quantile(1)



IQR = Q3 - Q1
alt_sinir = Q1 - 1.5 * IQR

ust_sinir = Q3 + 1.5 * IQR
print("Q1 ->", Q1)

print("Q3 ->", Q3)

print("Q2 ->", Q2)

print("Q4 ->", Q4)

print("IQR ->", IQR)



print()



print("Alt sınır: Q1 - 1.5 * IQR ->", alt_sinir)

print("Üst sınır: Q3 + 1.5 * IQR ->", ust_sinir)
outliers_df = df[(df["CozunurlukYükseklik"] < alt_sinir) | (df["CozunurlukYükseklik"] > ust_sinir)]

outliers_df
label_encoder = preprocessing.LabelEncoder()
df['Bluetooth'] = label_encoder.fit_transform(df['Bluetooth'])

df.head()
df['CiftHat'] = label_encoder.fit_transform(df['CiftHat'])

df.head()
df['4G'] = label_encoder.fit_transform(df['4G'])

df.head()
df['3G'] = label_encoder.fit_transform(df['3G'])

df.head()
df['Dokunmatik'] = label_encoder.fit_transform(df['Dokunmatik'])

df.head()
df['WiFi'] = label_encoder.fit_transform(df['WiFi'])

df.head()
df['Renk'] = pd.Categorical(df['Renk'])

dfDummies = pd.get_dummies(df['Renk'], prefix = 'Renk')

dfDummies
df = pd.concat([df, dfDummies], axis=1)

df.head()
df.drop(["Renk", "Renk_Kahverengi"], axis = 1, inplace = True)
df.head()
df.info()
df.sample(10)
df.head(10)
df["FiyatAraligi"].unique()
df['FiyatAraligi'] = label_encoder.fit_transform(df['FiyatAraligi'])

df.head()
df["FiyatAraligi"].unique()
df.corr()
X = df.drop("FiyatAraligi", axis = 1)

y = df["FiyatAraligi"]
X
y
X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y, 

                                                    test_size = 0.25, 

                                                    random_state = 35)

X_train
X_test
y_train
y_test
X_train
X_test
y_test
y_train
nb = GaussianNB()

nb_model = nb.fit(X_train, y_train)
nb_model
y_pred = nb_model.predict(X_test)
y_pred
y_test
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
(karmasiklik_matrisi[0][0] + karmasiklik_matrisi[1][1]+karmasiklik_matrisi[2][2] + karmasiklik_matrisi[3][3]) / (karmasiklik_matrisi[0][0] + karmasiklik_matrisi[0][1]+karmasiklik_matrisi[0][2]+karmasiklik_matrisi[0][3]+karmasiklik_matrisi[1][0] + karmasiklik_matrisi[1][1]+karmasiklik_matrisi[1][2]+karmasiklik_matrisi[1][3]+karmasiklik_matrisi[2][0] + karmasiklik_matrisi[2][1]+karmasiklik_matrisi[2][2]+karmasiklik_matrisi[2][3]+karmasiklik_matrisi[3][0] + karmasiklik_matrisi[3][1]+karmasiklik_matrisi[3][2]+karmasiklik_matrisi[3][3] )
cross_val_score(nb_model, X_test, y_test, cv = 12)
cross_val_score(nb_model, X, y, cv = 12).mean()
print(classification_report(y_test, y_pred))
cart = DecisionTreeClassifier(random_state = 42)

cart_model = cart.fit(X_train, y_train)
cart_model
y_pred = cart_model.predict(X_test)
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
(karmasiklik_matrisi[0][0] + karmasiklik_matrisi[1][1]+karmasiklik_matrisi[2][2] + karmasiklik_matrisi[3][3]) / (karmasiklik_matrisi[0][0] + karmasiklik_matrisi[0][1]+karmasiklik_matrisi[0][2]+karmasiklik_matrisi[0][3]+karmasiklik_matrisi[1][0] + karmasiklik_matrisi[1][1]+karmasiklik_matrisi[1][2]+karmasiklik_matrisi[1][3]+karmasiklik_matrisi[2][0] + karmasiklik_matrisi[2][1]+karmasiklik_matrisi[2][2]+karmasiklik_matrisi[2][3]+karmasiklik_matrisi[3][0] + karmasiklik_matrisi[3][1]+karmasiklik_matrisi[3][2]+karmasiklik_matrisi[3][3] )
cross_val_score(cart_model, X_test, y_test, cv = 12)
cross_val_score(cart_model, X_test, y_test, cv = 12).mean()
print(classification_report(y_test, y_pred))
cart = DecisionTreeClassifier(random_state = 42, criterion='entropy')

cart_model = cart.fit(X_train, y_train)
cart_model
y_pred = cart_model.predict(X_test)
accuracy_score(y_test, y_pred)
graph = Source(tree.export_graphviz(cart, out_file = None, feature_names = X.columns, filled = True))

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
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
accuracy_score(y_test, y_pred)
cross_val_score(knn_model, X_test, y_test, cv = 12)
cross_val_score(knn_model, X_test, y_test, cv = 12).mean()
print(classification_report(y_test, y_pred))
knn_params = {"n_neighbors": np.arange(2,15)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, knn_params, cv = 12)

knn_cv.fit(X_train, y_train)
print("En iyi skor: " + str(knn_cv.best_score_))

print("En iyi parametreler: " + str(knn_cv.best_params_))
score_list = []



for each in range(2,15,1):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(X_train,y_train)

    score_list.append(knn2.score(X_test, y_test))

    

plt.plot(range(2,15,1),score_list)

plt.xlabel("k değerleri")

plt.ylabel("doğruluk skoru")

plt.show()
knn = KNeighborsClassifier(14)

knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
knn = KNeighborsClassifier(10)

knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred)