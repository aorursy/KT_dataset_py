import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import missingno

from sklearn import preprocessing

import re

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
tablet = pd.read_csv("../input/tabletpc-priceclassification/tablet.csv")

df = tablet.copy() 
df.head()
df.shape
df.dtypes
df.isna().sum()
print(df["Renk"].unique())
print(df["FiyatAraligi"].unique())
df[df["CiftHat"] == "Var"]
df.groupby("FiyatAraligi").describe()["BataryaGucu"]
pd.pivot_table(df, values = "BataryaGucu", index = ["WiFi", "Bluetooth", "Dokunmatik", "4G"],columns = ["FiyatAraligi"], aggfunc = np.mean)
pd.pivot_table(df, values = ["BataryaGucu", "BataryaOmru"], index = ["WiFi", "Bluetooth", "Dokunmatik", "4G"],columns = ["FiyatAraligi"], aggfunc = np.mean)
pd.pivot_table(df, values = ["BataryaGucu", "BataryaOmru"], index = ["WiFi", "Bluetooth", "Dokunmatik", "4G","Renk"],columns = ["FiyatAraligi"], aggfunc = np.mean)
sns.distplot(df["CekirdekSayisi"], bins=8, color="red");
sns.countplot(df["Renk"]);
sns.scatterplot(x = "CozunurlukYükseklik", y = "CozunurlukGenislik", data = df);
sns.jointplot(x = df["CozunurlukYükseklik"], y = df["CozunurlukGenislik"], kind = "kde", color = "turquoise");
sns.scatterplot(x = "CozunurlukYükseklik", y = "CozunurlukGenislik", hue = "FiyatAraligi",  data = df);
sns.lmplot(x = "CozunurlukYükseklik", y = "CozunurlukGenislik", hue = "FiyatAraligi",  data = df);
sns.violinplot(x = "FiyatAraligi", y = "RAM",  data = df);
sns.countplot(df["FiyatAraligi"])
corr = df.corr()

corr
corr=df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
with sns.axes_style('dark'):

    sns.jointplot("ArkaKameraMP", "OnKameraMP", data = df, kind = "hex", color = "green")
df.corr()["ArkaKameraMP"]["OnKameraMP"]
df.describe()
df["BataryaOmru"].mean()
df[df["FiyatAraligi"] == "Ucuz"][["BataryaGucu","MikroislemciHizi","DahiliBellek","RAM"]].mean()

df[df["FiyatAraligi"] == "Çok Ucuz"][["OnKameraMP","ArkaKameraMP"]].max()
df["Agirlik"].min()
df.groupby("FiyatAraligi")["RAM"].mean()
missingno.matrix(df,figsize=(12, 10));
missingno.heatmap(df, figsize= (6,6));
eok = df[df["OnKameraMP"].isnull()]

eok
print("Ortalama RAM: ", df["RAM"].mean(),

      "\nBu tabletlerdeki ortalama RAM:", eok["RAM"].mean())
print("Ortalama batarya gücü: ", df["BataryaGucu"].mean(),", batarya ömrü: ",df["BataryaOmru"].mean() , 

      "\nBu tabletlerdeki ortalama batarya gücü:", eok["BataryaGucu"].mean(), ", batarya ömrü: ", eok["BataryaOmru"].mean())
print("Ortalama arka kamera MP'si: ", df["ArkaKameraMP"].mean(), 

      "\nBu tabletlerdeki ortalama arka kamera MP'si:", eok["ArkaKameraMP"].mean())
eokT = eok["OnKameraMP"].index

eokT
df.loc[eokT ,"OnKameraMP"] = 0
df.isna().sum()["OnKameraMP"]
df["OnKameraMP"].unique()
df[df["RAM"].isnull()]
print("Ortalama batarya gücü: ", df["BataryaGucu"].mean(),", batarya ömrü: ",df["BataryaOmru"].mean() , 

      "\nBu tabletlerdeki ortalama batarya gücü:", df[df["RAM"].isnull()]["BataryaGucu"].mean(), ", batarya ömrü: ", df[df["RAM"].isnull()]["BataryaOmru"].mean())
print("Ortalama mikro işlemci hızı: ", df["MikroislemciHizi"].mean(), 

      "\nBu tabletlerdeki ortalama mikro işlemci hızı:", df[df["RAM"].isnull()]["MikroislemciHizi"].mean())
print("Ortalama dahili bellek: ", df["DahiliBellek"].mean(), 

      "\nBu tabletlerdeki ortalama dahili bellek:", df[df["RAM"].isnull()]["DahiliBellek"].mean())
df[df["FiyatAraligi"] == "Pahalı"][["RAM"]].mean()
erT = df[df["RAM"].isnull()]["RAM"].index

erT
df.loc[erT, "RAM"] = 3449.35041
df["RAM"].isna().sum()
print(df["RAM"].unique())
label_encoder = preprocessing.LabelEncoder()
df['Bluetooth'] = label_encoder.fit_transform(df['Bluetooth']) 

df['CiftHat'] = label_encoder.fit_transform(df['CiftHat'])

df['4G'] = label_encoder.fit_transform(df['4G'])

df['3G'] = label_encoder.fit_transform(df['3G'])

df['Dokunmatik'] = label_encoder.fit_transform(df['Dokunmatik'])

df['WiFi'] = label_encoder.fit_transform(df['WiFi'])

df['FiyatAraligi'] = label_encoder.fit_transform(df['FiyatAraligi']) # Normal: 0, Pahalı: 1, Ucuz: 2, Çok Ucuz: 3 oldu.

df.head()
df['Renk'] = pd.Categorical(df['Renk'])

dfDummiesR = pd.get_dummies(df['Renk'])

dfDummiesR
df = pd.concat([df, dfDummiesR], axis=1)

df.drop(["Renk"], axis = 1, inplace = True)

df.drop(["Yeşil"], axis = 1, inplace = True)
df.head()
X = df.drop((["FiyatAraligi"]), axis=1)

Y = df["FiyatAraligi"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1/4, random_state = 20, shuffle=1)
model_GNB = GaussianNB()

model_GNB.fit(X_train, y_train)

y_pred_GNB = model_GNB.predict(X_test)
module_DT = DecisionTreeClassifier()

module_DT = module_DT.fit(X_train,y_train)

y_pred_DT = module_DT.predict(X_test)
module_DTE = DecisionTreeClassifier(criterion = "entropy")

module_DTE = module_DTE.fit(X_train,y_train)

y_pred_DTE = module_DTE.predict(X_test)
accuracy_DT = accuracy_score(y_test, y_pred_DT)

accuracy_DTE = accuracy_score(y_test, y_pred_DTE)

print("Varsayılan criterion ile başarı skoru: ", accuracy_DT, 

      " \ncriterion = entropy ile başarı skoru: ", accuracy_DTE)
y_pred_KNN_Score = []

for k in range(2,15):

    module_KNN = KNeighborsClassifier(n_neighbors = k)

    module_KNN = module_KNN.fit(X_train, y_train)

    y_pred_KNN = module_KNN.predict(X_test)

    y_pred_KNN_Score.append(accuracy_score(y_test, y_pred_KNN))
for k in range(2,15):

    print( "Komşu sayısı" , k , "iken modelin başarı skoru :" , y_pred_KNN_Score[k- 2])

    #k 2'den başlarken liste 0'dan başlar. Bu yüzden liste elemanı için k - 2 yazıyoruz.
plt.figure(figsize=(15, 6))

plt.plot(range(2, 15), y_pred_KNN_Score, color='gold', linestyle='-', marker=".",

         markerfacecolor='orange', markersize = 12)

plt.title('Komşu Sayısına Göre KNN Modelinin Başarı Skoru ')

plt.xlabel('Komşu Sayısı')

plt.ylabel('Başarı Skoru')
module_KNN = KNeighborsClassifier(n_neighbors = 13)

module_KNN = module_KNN.fit(X_train, y_train)

y_pred_KNN = module_KNN.predict(X_test)
GNB_ConMat = confusion_matrix(y_test,y_pred_GNB)

print(GNB_ConMat)
DTE_ConMat = confusion_matrix(y_test,y_pred_DTE)

print(DTE_ConMat)
KNN_ConMat = confusion_matrix(y_test,y_pred_KNN)

print(KNN_ConMat)
accuracy_GNB = accuracy_score(y_test, y_pred_GNB)

accuracy_KNN = accuracy_score(y_test, y_pred_KNN)

print("GaussianNB modelinin başarı skoru : ", accuracy_GNB, 

      "\nDecisionTree modelinin başarı skoru : ", accuracy_DTE, 

      "\nKNN modelinin başarı skoru : " , accuracy_KNN)
crGNB = classification_report(y_test, y_pred_GNB)

print(crGNB)
crDTE = classification_report(y_test, y_pred_DTE)

print(crDTE)
crKNN = classification_report(y_test, y_pred_KNN)

print(crKNN)