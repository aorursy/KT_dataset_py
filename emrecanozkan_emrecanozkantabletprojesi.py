import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
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
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import export_graphviz
from sklearn import tree
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
tablets = pd.read_csv("../input/tablet.csv", sep = ",")
tablets.head(n=10)
tablets.info()
tablets.shape
tablets.isnull().sum()
tablets["Bluetooth"].unique()
tablets["CiftHat"].unique()
tablets["OnKameraMP"].unique()
tablets["4G"].unique()
tablets["DahiliBellek"].unique()
tablets["Kalinlik"].unique()
tablets["Agirlik"].unique()
tablets["CekirdekSayisi"].unique()
tablets["ArkaKameraMP"].unique()
tablets["CozunurlukYükseklik"].unique()
tablets["CozunurlukGenislik"].unique()
tablets["RAM"].unique()
tablets["3G"].unique()
tablets["BataryaOmru"].unique()
tablets["Dokunmatik"].unique()
tablets["WiFi"].unique()
tablets["FiyatAraligi"].unique()
msno.matrix(tablets,figsize=(12,5))
dropedTablets = tablets.dropna()
dropedTablets
dropedTablets.corr()
sns.heatmap(dropedTablets.corr())
dropedTablets["FiyatAraligi"].unique()
sns.jointplot(x = dropedTablets["RAM"], y = dropedTablets["BataryaGucu"], kind = "kde", color = "purple");
sns.scatterplot(x = "RAM", y = "BataryaGucu", hue = "FiyatAraligi",  data = dropedTablets);
sns.scatterplot(x = "RAM", y = "CozunurlukYükseklik", hue = "FiyatAraligi",  data = dropedTablets);
sns.scatterplot(x = "RAM", y = "CozunurlukGenislik", hue = "FiyatAraligi",  data = dropedTablets);
sns.violinplot(y = "RAM", data = dropedTablets);
sns.violinplot(x = "RAM", y = "FiyatAraligi", data = dropedTablets);
sns.countplot(dropedTablets['FiyatAraligi'])
dropedTablets["RAM"].mean()
dropedTablets["RAM"].median()
dropedTablets["RAM"].std()
max(dropedTablets["RAM"])
min(dropedTablets["RAM"])
dropedTablets[(dropedTablets["RAM"] < 1000) & (dropedTablets["FiyatAraligi"] == "Pahalı")]
dropedTablets[(dropedTablets["RAM"] < 1000) & (dropedTablets["FiyatAraligi"] == "Normal")]
dropedTablets[(dropedTablets["RAM"] < 1000) & (dropedTablets["FiyatAraligi"] == "Ucuz")]
dropedTablets[(dropedTablets["RAM"] < 1000) & (dropedTablets["FiyatAraligi"] == "Ucuz")][["4G","WiFi","Dokunmatik","CiftHat","Bluetooth","FiyatAraligi"]]
dropedTablets[(dropedTablets["RAM"] > 1000) & (dropedTablets["FiyatAraligi"] == "Normal") & (dropedTablets["Dokunmatik"] == "Yok")][["4G","WiFi","Dokunmatik","CiftHat","Bluetooth","FiyatAraligi","RAM"]]
dropedTablets.groupby(["FiyatAraligi"]).mean()
dropedTablets.groupby(["FiyatAraligi"]).std()
dropedTablets.describe().T
newTablets = dropedTablets.drop("4G",axis=1)
tabletsDummies4G = pd.get_dummies(dropedTablets["4G"])
newTablets["4GVar"] = tabletsDummies4G["Var"]
newTablets["4GYok"] = tabletsDummies4G["Yok"]
del newTablets["3G"]
del newTablets["Bluetooth"]
del newTablets["CiftHat"]
del newTablets["Dokunmatik"]
del newTablets["WiFi"]
del newTablets["Renk"]
newTablets.head()
newTablets.describe().T
dropNadTablets = newTablets.dropna()
X = dropNadTablets.drop("FiyatAraligi",axis=1)
y = dropNadTablets["FiyatAraligi"]
X
y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train
X_test
clf = GaussianNB()
gaussianNBTrain = clf.fit(X,y)
y_pred = clf.predict(X_test)
accuracy_score(y_test,y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)
karmasiklik_matrisi
cross_val_score(gaussianNBTrain,X,y,cv = 10)
cross_val_score(gaussianNBTrain,X,y,cv = 10).mean()
print(classification_report(y_test, y_pred))
cart = DecisionTreeClassifier(random_state = 42, criterion='gini')
cart_model = cart.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
accuracy_score(y_test,y_pred)
cart = DecisionTreeClassifier(random_state = 42, criterion='entropy')
cart_model = cart.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
accuracy_score(y_test,y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)
karmasiklik_matrisi
cross_val_score(cart_model, X, y, cv = 10)
cross_val_score(cart_model, X, y, cv = 10).mean()
print(classification_report(y_test, y_pred))
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
komsuSayilarinaGoreSkorArray = []
komsuSayilarinaGoreSkorArraySira = [2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0]
knn = KNeighborsClassifier(n_neighbors=15)
knn_model = knn.fit(X_train, y_train)
y_pred = knn_model.predict(X)
komsuSayilarinaGoreSkorArray.append(accuracy_score(y,y_pred))
accuracy_score(y, y_pred)
sns.scatterplot(x = komsuSayilarinaGoreSkorArraySira, y = komsuSayilarinaGoreSkorArray, color = "blue");
karmasiklik_matrisi = confusion_matrix(y, y_pred)
print(karmasiklik_matrisi)
cross_val_score(knn_model, X, y, cv = 10)
cross_val_score(knn_model, X, y, cv = 10).mean()
print(classification_report(y, y_pred))