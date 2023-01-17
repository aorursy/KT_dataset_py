import numpy as np

import seaborn as sns

import pandas as pd

import missingno           



from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plot 

import os

from subprocess import check_output

from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, precision_score

from sklearn.naive_bayes import GaussianNB

from sklearn import preprocessing

from warnings import filterwarnings

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix as cm

from sklearn import ensemble

from sklearn.tree import export_graphviz

from sklearn import tree

from IPython.display import SVG

from graphviz import Source

from IPython.display import display

filterwarnings('ignore')



sns.set(rc={'figure.figsize':(10,8)})
df = pd.read_csv('../input/tabletpc-priceclassification/tablet.csv')
df
df.head()
df.shape
df.columns
df.info()
df.isnull().sum()
df.describe()
df["FiyatAraligi"].unique()
df.FiyatAraligi.value_counts()
sns.countplot(x = "FiyatAraligi", data = df);
df.hist()
sns.scatterplot(x = "CozunurlukGenislik", y = "CozunurlukYükseklik", hue = "FiyatAraligi", data = df); #Çözünürlük genişlik ve yükseklik arasında da genel olarak doğrusal bir ilişki var.
sns.scatterplot(x = "ArkaKameraMP", y = "OnKameraMP", hue = "FiyatAraligi", data = df); #Örneğin ön kamera ve arka kamera arasındaki fiyat aralığı dağılımına bakalım. Değerler üçgen şeklinde artarak seyrediyor. Arka kamera çözünürlüğü arttıkça ön kamera çözünürlüğü de artıyor. Doğru orantısız değerler de var.
sns.scatterplot(x = "BataryaGucu", y = "BataryaOmru", hue = "FiyatAraligi", data = df); 
sns.distplot(df["RAM"], bins=32, color="red");
sns.violinplot(x = "FiyatAraligi", y = "RAM", data = df);
sns.relplot(x='OnKameraMP', y='ArkaKameraMP', hue='FiyatAraligi', size='FiyatAraligi', col='FiyatAraligi', data=df)
missingno.matrix(df,figsize=(20, 10)); 
missingno.heatmap(df, figsize= (20,10));
df.groupby("FiyatAraligi").mean()
df.groupby("FiyatAraligi")[["OnKameraMP"]].mean() 
sns.countplot(df[df["OnKameraMP"].isnull()]["FiyatAraligi"]);
cok_ucuz_OnKameraMP = df[(df["FiyatAraligi"] == "Çok Ucuz") & (df["OnKameraMP"].isnull())].index

cok_ucuz_OnKameraMP
df.loc[cok_ucuz_OnKameraMP,"OnKameraMP"] = 4
df.isna().sum()["OnKameraMP"]
df.groupby("FiyatAraligi")[["RAM"]].mean() 
sns.countplot(df[df["RAM"].isnull()]["FiyatAraligi"]);
df[(df["FiyatAraligi"] == "Pahalı") & (df["RAM"].isnull())]
pahali_RAM = df[(df["FiyatAraligi"] == "Pahalı") & (df["RAM"].isnull())].index

pahali_RAM
df.loc[pahali_RAM,"RAM"] = 3450
df.isna().sum()["RAM"]
le = preprocessing.LabelEncoder()

df["Bluetooth"] = le.fit_transform(df["Bluetooth"])

df.head()
df["CiftHat"] = le.fit_transform(df["CiftHat"])

df.head()
df["4G"] = le.fit_transform(df["4G"])

df.head()
df["Dokunmatik"] = le.fit_transform(df["Dokunmatik"])

df.head()
df["3G"] = le.fit_transform(df["3G"])

df.head()
df["WiFi"] = le.fit_transform(df["WiFi"])

df.head()
df["Renk_Encoded"] = le.fit_transform(df["Renk"])

df.head()
df["FiyatAraligi_Encoded"] =  le.fit_transform(df["FiyatAraligi"])

df.head()
df.groupby(["FiyatAraligi"]).mean()
df.groupby(["FiyatAraligi"]).std() 
df.corr()
df.corr()["ArkaKameraMP"]["OnKameraMP"]
corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) 

 

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] 

    columnNames = list(df)

    if len(columnNames) > 10: 

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plot.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plot.suptitle('Scatter and Density Plot')

    plot.show()

plotScatterMatrix(df,20,10)
lineer_regresyon = LinearRegression() 
X = df.drop(["FiyatAraligi_Encoded","FiyatAraligi","Renk"], axis = 1)

y = df["FiyatAraligi_Encoded"] 
X
y
lineer_regresyon.fit(X, y)
lineer_regresyon.predict([[1233,1,1,1,2,0,50,0.1,24,1,3,428,695,2000,2,0,1,1,0,]]) 
df["model1_prediction"] = lineer_regresyon.predict(X)

df
X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size = 0.25, 

                                                    random_state = 42)
X_train
X_test
y_train
y_test
NB = GaussianNB()

NB_model = NB.fit(X_train,y_train)
NB_model
X_test[0:20]
NB_model.predict(X_test)[0:20]
y_test[0:20]
y_pred = NB_model.predict(X_test)
y_pred
y_test
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

karmasiklik_matrisi
(karmasiklik_matrisi[0][0] + karmasiklik_matrisi[1][1]) / (karmasiklik_matrisi[0][0] + karmasiklik_matrisi[1][1] +  karmasiklik_matrisi[1][0] + karmasiklik_matrisi[0][1])
cross_val_score(NB_model, X_test, y_test, cv = 20)
cross_val_score(NB_model, X_test, y_test, cv = 20).mean()
print(classification_report(y_test, y_pred))
PrecisionScore = precision_score(y_test, y_pred, average='weighted')

PrecisionScore
F1Score = f1_score(y_test, y_pred, average = 'weighted')  

F1Score
RecallScore = recall_score(y_test, y_pred, average='weighted')

RecallScore
cart = DecisionTreeClassifier(random_state = 42, criterion='gini')

cart_model_gini = cart.fit(X_train, y_train)
cart_model_gini
y_pred_gini = cart_model_gini.predict(X_test)
accuracy_score(y_test, y_pred_gini)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred_gini)

print(karmasiklik_matrisi)
cross_val_score(cart_model_gini, X, y, cv = 20)
cross_val_score(cart_model_gini, X, y, cv = 20).mean()
print(classification_report(y_test, y_pred_gini))
cart = DecisionTreeClassifier(random_state = 42, criterion='entropy')

cart_model = cart.fit(X_train, y_train)
cart_model
y_pred2 = cart_model.predict(X_test)
accuracy_score(y_test, y_pred2)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred2)

print(karmasiklik_matrisi)
cross_val_score(cart_model, X, y, cv = 20)
cross_val_score(cart_model, X, y, cv = 20).mean()
print(classification_report(y_test, y_pred2))
graph = Source(tree.export_graphviz(cart, out_file = None, feature_names = X.columns, filled = True))

display(SVG(graph.pipe(format = 'svg')))
ranking = cart.feature_importances_

features = np.argsort(ranking)[::-1][:20]

columns = X.columns



plot.figure(figsize = (16, 9))

plot.title("Karar Ağacına Göre Özniteliklerin Önem Derecesi", y = 1.03, size = 18)

plot.bar(range(len(features)), ranking[features], color="purple", align="center")

plot.xticks(range(len(features)), columns[features], rotation=80)

plot.show()
knn = KNeighborsClassifier()

knn_model = knn.fit(X_train, y_train)
knn_model
y_pred3 = knn_model.predict(X)
accuracy_score(y, y_pred3)
karmasiklik_matrisi = confusion_matrix(y, y_pred3)

print(karmasiklik_matrisi)
cross_val_score(knn_model, X_test, y_test, cv = 20)
cross_val_score(knn_model, X_test, y_test, cv = 20).mean()
print(classification_report(y, y_pred3))
knn_params = {"n_neighbors": np.arange(2,15)}

knn_params
knn_komsu = KNeighborsClassifier()

knn_cv = GridSearchCV(knn_komsu, knn_params, cv = 3)

knn_cv.fit(X_train, y_train)
print("En iyi skor: " + str(knn_cv.best_score_))

print("En iyi parametreler: " + str(knn_cv.best_params_))
knn = KNeighborsClassifier(9)

knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test) 
accuracy_score(y_test, y_pred)
skor_listesi = []



for each in range(2,15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(X_train,y_train)

    skor_listesi.append(knn2.score(X_test, y_test))

    

plot.plot(range(2,15),skor_listesi)

plot.xlabel("k değerleri")

plot.ylabel("doğruluk skoru")

plot.show()
cross_val_score(knn_tuned, X_test, y_test, cv = 20)

cross_val_score(knn_tuned, X_test, y_test, cv = 20).mean()