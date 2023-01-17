#Gerekli kütüphanelerimizi ekliyoruz.



import numpy as np

import pandas as pd 



from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score, cross_val_predict

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 

from sklearn import model_selection

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import BaggingRegressor

from argparse import Namespace

import seaborn as sns 

from sklearn.feature_selection import RFE 





from sklearn import preprocessing

from warnings import filterwarnings

filterwarnings('ignore')





sns.set(rc={'figure.figsize':(10,8)})
tablet = pd.read_csv("../input/tabletpc-priceclassification/tablet.csv")

df = tablet.copy()
df.head()
df.info()
df.shape
df.isnull().sum()
df.dtypes
df.describe().T
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
df["BataryaGucu"].value_counts()
df["Bluetooth"].value_counts()
df["MikroislemciHizi"].value_counts()
df["CiftHat"].value_counts()
df["OnKameraMP"].value_counts()
df["DahiliBellek"].value_counts()
df["Kalinlik"].value_counts()
df["Agirlik"].value_counts()
df["CekirdekSayisi"].value_counts()
df["ArkaKameraMP"].value_counts()
df["CozunurlukYükseklik"].value_counts()
df["CozunurlukGenislik"].value_counts()
df["RAM"].value_counts()
df["BataryaOmru"].value_counts()
df["FiyatAraligi"].value_counts()
df["3G"].value_counts()
df["Renk"].value_counts()
df["WiFi"].value_counts()
df["Dokunmatik"].value_counts()
df["BataryaGucu"].describe()
sns.violinplot(df["BataryaGucu"]);
sns.distplot(df["BataryaGucu"], bins = 20);
sns.boxplot(df["BataryaGucu"]);
df["Bluetooth"].describe()
sns.countplot(df["Bluetooth"]);
df["MikroislemciHizi"].describe()
sns.violinplot(df["MikroislemciHizi"]);
sns.boxplot(df["MikroislemciHizi"]);
df["CiftHat"].describe()
sns.countplot(df["CiftHat"]);
df["OnKameraMP"].describe()
sns.violinplot(df["OnKameraMP"]);
sns.distplot(df["OnKameraMP"], bins = 20);
sns.boxplot(df["OnKameraMP"]);
df["DahiliBellek"].describe()
sns.violinplot(df["DahiliBellek"]);
sns.boxplot(df["DahiliBellek"]);
df["Kalinlik"].describe()
sns.violinplot(df["Kalinlik"]);
sns.boxplot(df["Kalinlik"]);
df["Agirlik"].describe()
sns.violinplot(df["Agirlik"]);
sns.boxplot(df["Agirlik"]);
df["CekirdekSayisi"].describe()
sns.violinplot(df["CekirdekSayisi"]);
sns.boxplot(df["CekirdekSayisi"]);
sns.countplot(df["CekirdekSayisi"]);
df["ArkaKameraMP"].describe()
sns.violinplot(df["ArkaKameraMP"]);
sns.boxplot(df["ArkaKameraMP"]);
df["CozunurlukYükseklik"].describe()
sns.violinplot(df["CozunurlukYükseklik"]);
sns.distplot(df["CozunurlukYükseklik"], bins = 20);
sns.boxplot(df["CozunurlukYükseklik"]);
df["CozunurlukGenislik"].describe()
sns.violinplot(df["CozunurlukGenislik"]);
sns.distplot(df["CozunurlukGenislik"], bins = 20);
sns.boxplot(df["CozunurlukGenislik"]);
df["RAM"].describe()
sns.violinplot(df["RAM"]);
sns.distplot(df["RAM"], bins = 20);
sns.boxplot(df["RAM"]);
df["BataryaOmru"].describe()
sns.violinplot(df["BataryaOmru"]);
sns.boxplot(df["BataryaOmru"]);
sns.catplot(x = "FiyatAraligi", y = "BataryaOmru", data = df, height = 20, alpha = .5);
sns.violinplot(x = "FiyatAraligi", y = "BataryaOmru", data = df, height = 15, alpha = .5);
sns.catplot(x = "FiyatAraligi", y = "BataryaGucu", data = df, height = 15, alpha = .5);
sns.violinplot(x = "FiyatAraligi", y = "BataryaGucu", data = df, height = 15, alpha = .5);
sns.catplot(x = "FiyatAraligi", y = "MikroislemciHizi", data = df, height = 15, alpha = .5);
sns.violinplot(x = "FiyatAraligi", y = "MikroislemciHizi", data = df, height = 15, alpha = .5);
sns.catplot(x = "FiyatAraligi", y = "OnKameraMP", data = df, height = 15, alpha = .5);
sns.violinplot(x = "FiyatAraligi", y = "OnKameraMP", data = df, height = 15, alpha = .5);
sns.catplot(x = "FiyatAraligi", y = "DahiliBellek", data = df, height = 15, alpha = .5);
sns.violinplot(x = "FiyatAraligi", y = "DahiliBellek", data = df, height = 8 ,alpha = .5);
sns.catplot(x = "FiyatAraligi", y = "CekirdekSayisi", data = df, height = 12, alpha = .1);
sns.violinplot(x = "FiyatAraligi", y = "CekirdekSayisi", data = df, height = 8, alpha = .5);
sns.catplot(x = "FiyatAraligi", y = "ArkaKameraMP", data = df, height = 15, alpha = .5);
sns.violinplot(x = "FiyatAraligi", y = "ArkaKameraMP", data = df, height = 8 ,alpha = .5);
sns.catplot(x = "FiyatAraligi", y = "CozunurlukYükseklik", data = df, height = 15, alpha = .5);
sns.violinplot(x = "FiyatAraligi", y = "CozunurlukYükseklik", data = df, height = 8, alpha = .5);
sns.catplot(x = "FiyatAraligi", y = "CozunurlukGenislik", data = df, height = 15, alpha = .5);
sns.violinplot(x = "FiyatAraligi", y = "CozunurlukGenislik", data = df, height = 8 ,alpha = .5);
sns.catplot(x = "FiyatAraligi", y = "RAM", data = df, height = 15, alpha = .5);
sns.violinplot(x = "FiyatAraligi", y = "RAM", data = df, height = 8, alpha = .5);
sns.scatterplot("FiyatAraligi", "MikroislemciHizi", "RAM", data = df);
sns.scatterplot("FiyatAraligi", "CozunurlukYükseklik", "CozunurlukGenislik", data = df);
sns.scatterplot("FiyatAraligi", "BataryaGucu", "BataryaOmru", data = df);
sns.scatterplot("FiyatAraligi", "ArkaKameraMP", "OnKameraMP", data = df);
corr = df.corr()

corr
sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
df.hist(figsize = (15,15))

plt.show()
import missingno # eksik verileri daha iyi okumak için missingno kütüphanesini ekliyoruz.

missingno.matrix(df,figsize=(20, 10));
missingno.heatmap(df, figsize= (8,8));

#heatmap büyük veriler için faydalı olabilecek bir yöntem . burada eksik veriler arasında veri güçlü bir ilişki yok
def eksik_deger_tablosu(df): 

    eksik_deger = df.isnull().sum()

    eksik_deger_yuzde = 100 * df.isnull().sum()/len(df)

    eksik_deger_tablo = pd.concat([eksik_deger, eksik_deger_yuzde], axis=1)

    eksik_deger_tablo_son = eksik_deger_tablo.rename(columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})

    return eksik_deger_tablo_son
eksik_deger_tablosu(df)
df["RAM"].unique()
df["OnKameraMP"].unique()
esikDeger = len(df) * .3
df.dropna(thresh = esikDeger, axis = 1, inplace = True)
df.isna().sum()
df[df["OnKameraMP"].isnull()]
eksikOnKameraMP = df[df["OnKameraMP"].isnull()].index

eksikOnKameraMP
df.loc[eksikOnKameraMP ,"OnKameraMP"] = 1
df[df["RAM"].isnull()]
eksikRAM = df[df["RAM"].isnull()].index

eksikRAM
df.loc[eksikRAM ,"RAM"] = 1
label_encoder = preprocessing.LabelEncoder()
df['Bluetooth'] = label_encoder.fit_transform(df['Bluetooth'])
df['4G'] = label_encoder.fit_transform(df['4G'])
df['CiftHat'] = label_encoder.fit_transform(df['CiftHat'])
df['Dokunmatik'] = label_encoder.fit_transform(df['Dokunmatik'])
df['WiFi'] = label_encoder.fit_transform(df['WiFi'])
df['3G'] = label_encoder.fit_transform(df['3G'])
df['Renk'] = pd.Categorical(df['Renk'])

dfRenkKategorileri = pd.get_dummies(df['Renk'], prefix = 'Renk')

dfRenkKategorileri
df = pd.concat([df,dfRenkKategorileri],axis=1)

df.head()
df.drop(["Renk"], axis = 1, inplace = True)

df.head()
df.drop(["Renk_Siyah"], axis = 1, inplace = True)

df.head()
X = df.drop(["FiyatAraligi"], axis=1)
y = df["FiyatAraligi"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
X
y
X_train
X_test
y_train
y_test
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, precision_score

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

nb_model = nb.fit(X_train, y_train)

nb_model
y_pred = nb_model.predict(X_test)
from sklearn.metrics import  accuracy_score

accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix#

karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier

cart = DecisionTreeClassifier(random_state = 42)

cart_model = cart.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
accuracy_score(y_test, y_pred)
cart = DecisionTreeClassifier(random_state = 42, criterion='entropy')
cart_model = cart.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
print(classification_report(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn_model = knn.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
print(classification_report(y_test, y_pred))
knn_params = {"n_neighbors": np.arange(1,20)}
from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, knn_params, cv = 3)

knn_cv.fit(X_train, y_train)
print("En iyi skor: " + str(knn_cv.best_score_))
print("En iyi parametreler: " + str(knn_cv.best_params_))
armasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
print(classification_report(y_test, y_pred))
score_list = []

for each in range(1,20,1):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(X_train,y_train)

    score_list.append(knn2.score(X_test, y_test))



plt.plot(range(1,20,1),score_list)

plt.xlabel("k en yakın komşu sayıları")

plt.ylabel("doğruluk skoru")
knn = KNeighborsClassifier(11)

knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
print(classification_report(y_test, y_pred))