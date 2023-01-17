import numpy as np
import pandas as pd
import math
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve, recall_score, f1_score, precision_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/tablet.csv')
df.sample(10)
df.shape
df.info()
df.describe().T
df.corr()
sns.heatmap(df.corr())
df["Renk"].unique()
df["Renk"].nunique()
df["4G"].unique()
df["3G"].unique()
df["Bluetooth"].unique()
df["Dokunmatik"].unique()
df["CiftHat"].unique()
df["WiFi"].unique()
sns.countplot(df["WiFi"])
df["WiFi"].value_counts()
sns.countplot(df["FiyatAraligi"])
df["FiyatAraligi"].value_counts()
sns.countplot(df["Dokunmatik"])
df["Dokunmatik"].value_counts()
sns.countplot(df["4G"])
df["4G"].value_counts()
sns.countplot(df["CiftHat"])
df["CiftHat"].value_counts()
sns.countplot(df["3G"])
df["3G"].value_counts()
sns.countplot(df["Renk"])
df["Renk"].value_counts()
sns.countplot(df["Bluetooth"])
df["Bluetooth"].value_counts()
sns.scatterplot(y = "OnKameraMP", x = "ArkaKameraMP", hue = "FiyatAraligi",data = df)
sns.scatterplot(x = "CozunurlukGenislik", y = "CozunurlukYükseklik", hue= "FiyatAraligi", data = df)
sns.barplot(x = "FiyatAraligi", y = "RAM", data = df)
sns.barplot(x = "FiyatAraligi", y = "BataryaGucu", data = df)
sns.barplot(x = "FiyatAraligi", y = "BataryaOmru", data = df)
sns.barplot(x = "FiyatAraligi", y = "OnKameraMP", data = df)
sns.barplot(x = "FiyatAraligi", y = "ArkaKameraMP", data = df)
sns.barplot(x = "FiyatAraligi", y = "Kalinlik", data = df)
sns.barplot(x = "FiyatAraligi", y = "CozunurlukGenislik", data = df)
sns.barplot(x = "FiyatAraligi", y = "CozunurlukYükseklik", data = df)
sns.barplot(x = "FiyatAraligi", y = "Agirlik", data = df)
sns.barplot(x = "FiyatAraligi", y = "MikroislemciHizi", data = df)
sns.barplot(x = "FiyatAraligi", y = "CekirdekSayisi", data = df)
sns.barplot(x = "FiyatAraligi", y = "DahiliBellek", data = df)
def SinirlariYazdir(Q1,Q2,Q3,Q4,IQR):
    print("Q1-->", Q1)
    print("Q3-->", Q3)
    print("Q2-->", Q2)
    print("Q4-->", Q4)
    print("IQR-->", IQR)
    altsinir = Q1 - 1.5 * IQR
    ustsinir = Q3 + 1.5 * IQR
    print("Alt sınır: Q1 - 1.5 * IQR--->", altsinir)
    print("Üst sınır: Q3 + 1.5 * IQR--->", ustsinir)

def SinirlariBul(oznitelik):
    outliers= []
    Q1 = df[oznitelik].quantile(0.25)
    Q2 = df[oznitelik].quantile(0.5)
    Q3 = df[oznitelik].quantile(0.75)
    Q4 = df[oznitelik].quantile(1)
    IQR = Q3 - Q1
    print("{0} özniteliği için genel değerler".format(oznitelik))
    SinirlariYazdir(Q1,Q2,Q3,Q4,IQR)

sns.boxplot(x = "RAM", data = df)
SinirlariBul("RAM")
sns.boxplot(x = "BataryaGucu", data = df)
SinirlariBul("BataryaGucu")
sns.boxplot(x = "BataryaOmru", data = df)
SinirlariBul("BataryaOmru")
sns.boxplot(x = "OnKameraMP", data = df)
SinirlariBul("OnKameraMP")
df[df.OnKameraMP > 16]
sns.boxplot(x = "ArkaKameraMP", data = df)
SinirlariBul("ArkaKameraMP")
sns.boxplot(x = "Kalinlik", data = df)
SinirlariBul("Kalinlik")
sns.boxplot(x = "CozunurlukYükseklik", data = df)
SinirlariBul("CozunurlukYükseklik")
df[df.CozunurlukYükseklik > 1944]
df[df["CozunurlukYükseklik"] == 0]
df[df.FiyatAraligi == "Ucuz"]["CozunurlukGenislik"].mean() / df[df.FiyatAraligi == "Ucuz"]["CozunurlukYükseklik"].mean()
df.iloc[662, df.columns.get_loc('CozunurlukYükseklik')] = df.iloc[662].CozunurlukGenislik / 1.8742465046814174
df[df.FiyatAraligi == "Pahalı"]["CozunurlukGenislik"].mean() / df[df.FiyatAraligi == "Ucuz"]["CozunurlukYükseklik"].mean()
df.iloc[856, df.columns.get_loc('CozunurlukYükseklik')] = df.iloc[856].CozunurlukGenislik / 2.0481037409355634
sns.boxplot(x = "CozunurlukGenislik", data = df)
SinirlariBul("CozunurlukGenislik")
sns.boxplot(x = "Agirlik", data = df)
SinirlariBul("Agirlik")
sns.boxplot(x = "CekirdekSayisi", data = df)
SinirlariBul("CekirdekSayisi")
sns.boxplot(x = "MikroislemciHizi", data = df)
SinirlariBul("MikroislemciHizi")
sns.boxplot(x = "DahiliBellek", data = df)
SinirlariBul("DahiliBellek")
df.isnull().sum()
df[df["OnKameraMP"].isnull()]
df[df["ArkaKameraMP"] == 0]["OnKameraMP"]
df[df["ArkaKameraMP"] == 0].count()
df[(df["OnKameraMP"] == 0) & (df["ArkaKameraMP"] == 0)].count()
df.iloc[792, df.columns.get_loc('OnKameraMP')] = 0
df[df["OnKameraMP"].isnull()]
df[(df["ArkaKameraMP"] >= 12) & (df["ArkaKameraMP"] <= 16) & 
    (df["RAM"] > 900) & (df["RAM"] < 1200)]["OnKameraMP"].mean()
df.iloc[1641, df.columns.get_loc('OnKameraMP')] = 6
df[df["OnKameraMP"].isnull()]
df[(df["ArkaKameraMP"] >= 7) & (df["ArkaKameraMP"] <= 11) & 
    (df["RAM"] > 400) & (df["RAM"] < 600)]["OnKameraMP"].mean()
df[(df["ArkaKameraMP"] >= 1) & (df["ArkaKameraMP"] <= 5) & 
    (df["RAM"] > 400) & (df["RAM"] < 600)]["OnKameraMP"].mean()
df.iloc[726, df.columns.get_loc('OnKameraMP')] = 1
df.iloc[1416, df.columns.get_loc('OnKameraMP')] = 4
df[df["OnKameraMP"].isnull()]
df[(df["ArkaKameraMP"] >= 18) & (df["ArkaKameraMP"] <= 22) & 
    (df["RAM"] > 1200) & (df["RAM"] < 1450)]["OnKameraMP"].mean()
df.iloc[351, df.columns.get_loc('OnKameraMP')] = 9
df[df["RAM"].isnull()]
df[df["FiyatAraligi"] == "Pahalı"]["RAM"].mean()
df.fillna(3449, inplace=True)
ayni_olan_satirlar =[] 
 
for satir in df.itertuples(): 
    for aranacak_satir in df.itertuples():
        
        if (satir[1] == aranacak_satir[1] and satir[2] == aranacak_satir[2] and satir[3] == aranacak_satir[3] and 
            satir[4] == aranacak_satir[4] and satir[5] == aranacak_satir[5] and satir[6] == aranacak_satir[6] and
            satir[7] == aranacak_satir[7] and satir[8] == aranacak_satir[8] and satir[9] == aranacak_satir[9] and
            satir[10] == aranacak_satir[10] and satir[11] == aranacak_satir[11] and satir[12] == aranacak_satir[12] and
            satir[13] == aranacak_satir[13] and satir[14] == aranacak_satir[14] and satir[15] == aranacak_satir[15] and
            satir[16] == aranacak_satir[16] and satir[17] == aranacak_satir[17] and satir[18] == aranacak_satir[18] and
            satir[19] == aranacak_satir[19] and satir[20] != aranacak_satir[20]):
            ayni_olan_satirlar.append([satir[0],aranacak_satir[0]]) 
        
print(ayni_olan_satirlar)
sns.barplot(x = "RAM", y = "Renk", data = df)
df.drop(columns = "Renk", axis = 1, inplace = True)
def degerleri_duzelt(deger):
    if deger == "Var":
        return 1
    else:
        return 0
df["3G"] = df["3G"].apply(degerleri_duzelt)
df["4G"] = df["4G"].apply(degerleri_duzelt)
df["WiFi"] = df["WiFi"].apply(degerleri_duzelt)
df["CiftHat"] = df["CiftHat"].apply(degerleri_duzelt)
df["Bluetooth"] = df["Bluetooth"].apply(degerleri_duzelt)
df["Dokunmatik"] = df["Dokunmatik"].apply(degerleri_duzelt)
df.head()
X = df.drop("FiyatAraligi", axis = 1)
y = df["FiyatAraligi"]
X
y
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, 
                                                    test_size = 0.25)
X_train
X_test
y_train
y_test
nb = GaussianNB()
nb_model = nb.fit(X_train, y_train)
nb_model
y_pred = nb_model.predict(X_test)
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)
karmasiklik_matrisi
y_test.value_counts()
cart = DecisionTreeClassifier()
cart_model_gini = cart.fit(X_train, y_train)
y_pred_gini = cart_model_gini.predict(X_test)
accuracy_score(y_test, y_pred_gini)
karmasiklik_matrisi_gini = confusion_matrix(y_test, y_pred_gini)
karmasiklik_matrisi_gini
y_test.value_counts()
cart = DecisionTreeClassifier(criterion='entropy')
cart_model_entropy = cart.fit(X_train, y_train)
y_pred_entropy = cart_model_entropy.predict(X_test)
accuracy_score(y_test, y_pred_entropy)
karmasiklik_matrisi_entropy = confusion_matrix(y_test, y_pred_entropy)
karmasiklik_matrisi_entropy
knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
accuracy_score(y_test, y_pred_knn)
karmasiklik_matrisi_knn = confusion_matrix(y_test, y_pred_knn)
karmasiklik_matrisi
knn_params = {"n_neighbors": np.arange(2,15)}
knn_cv = GridSearchCV(knn, knn_params, cv = 5)
knn_cv.fit(X_train, y_train)
print("En iyi skor: " + str(knn_cv.best_score_))
print("En iyi parametreler: " + str(knn_cv.best_params_))
knn_cv.best_params_
komsu_sayilari = knn_cv.cv_results_["param_n_neighbors"].data
skorlar = knn_cv.cv_results_["mean_test_score"]
plt.figure(figsize=(10,6.8))
sns.barplot(x = komsu_sayilari, y = skorlar)
knn_best = KNeighborsClassifier(n_neighbors = knn_cv.best_params_["n_neighbors"])
knn_model_best = knn_best.fit(X_train, y_train)
y_pred_best = knn_model_best.predict(X_test)
accuracy_score(y_test, y_pred_best)
karmasiklik_matrisi_knn_best = confusion_matrix(y_test, y_pred_best)
karmasiklik_matrisi_knn_best
print(classification_report(y_test, y_pred_best))
cross_val_score(knn_model_best, X, y, cv = 5).mean()