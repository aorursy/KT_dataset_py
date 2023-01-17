import pandas as pd

import numpy as np

import seaborn as sns

import missingno as msno

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve,classification_report,f1_score, accuracy_score, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve,auc

from sklearn.model_selection import cross_val_predict
df=pd.read_csv("../input/tabletcsv/tablet.csv")

df.head()
df._get_numeric_data()
df.ndim
df.shape
df.size
df.info()
df.describe().T
df.corr()
df.cov()
sns.heatmap(df.corr())
df["FiyatAraligi"].value_counts()
df["FiyatAraligi"].unique()
df.FiyatAraligi.value_counts().plot(kind = 'barh')
df.isnull().sum()
msno.bar(df)
msno.matrix(df)
msno.heatmap(df)
df[df.isnull().any(axis=1)]
df[df["RAM"].isnull()]
mean = df["RAM"].mean()

df['RAM'].fillna(mean, inplace=True)
df[df["OnKameraMP"].isnull()]
median = df["OnKameraMP"].median()

df['OnKameraMP'].fillna(median, inplace=True)
df.isnull().sum().sum()
msno.matrix(df)
df.columns
sns.violinplot(x="FiyatAraligi",y="BataryaGucu",data=df)
df["BataryaGucu"].describe()
sns.boxplot(df["BataryaGucu"])
sns.distplot( df["BataryaGucu"], bins=20 )
sns.kdeplot(df['BataryaGucu'], shade=True)

df["Bluetooth"].unique()
df["Bluetooth"]=df["Bluetooth"].map({"Var":1,"Yok":0})
df["Bluetooth"].head()
sns.violinplot(x="FiyatAraligi",y="Bluetooth",data=df)
sns.violinplot(x="FiyatAraligi",y="MikroislemciHizi",data=df)
sns.boxplot(df["MikroislemciHizi"])
sns.distplot( df["MikroislemciHizi"], bins=20 )
sns.kdeplot(df['MikroislemciHizi'], shade=True)
df["CiftHat"].unique()
df["CiftHat"]=df["CiftHat"].map({"Var":1,"Yok":0})
df["CiftHat"].head()
sns.violinplot(x="FiyatAraligi",y="CiftHat",data=df)
sns.violinplot(x="FiyatAraligi",y="OnKameraMP",data=df)
sns.boxplot(df["OnKameraMP"])
Q1=df["OnKameraMP"].quantile(0.25)

Q3=df["OnKameraMP"].quantile(0.75)

IQR=Q3-Q1

altsinir_onkamera=Q1-1.5*IQR

ustsinir_onkamera=Q3+1.5*IQR

print("AltSınır:{0}\nÜstSınır:{1}".format(altsinir_onkamera,ustsinir_onkamera))
dfOnKameraMP=df["OnKameraMP"]

aykiri_OnKameraMP=((dfOnKameraMP<0)|(dfOnKameraMP>16.0))

dfOnKameraMP[aykiri_OnKameraMP]=16.0
sns.boxplot(df["OnKameraMP"])
sns.distplot(df["OnKameraMP"],bins=20)
sns.kdeplot(df['OnKameraMP'], shade=True)
sns.violinplot(x="FiyatAraligi",y="ArkaKameraMP",data=df)
sns.boxplot(df["ArkaKameraMP"])
sns.distplot(df["ArkaKameraMP"],bins=20)
sns.kdeplot(df["ArkaKameraMP"],shade=True)
sns.scatterplot(x = df["ArkaKameraMP"], y = df["OnKameraMP"])
sns.jointplot(x = df["ArkaKameraMP"], y = df["OnKameraMP"])
sns.lmplot(x="ArkaKameraMP",y="OnKameraMP",data=df,col="FiyatAraligi",scatter_kws={"color":"green"},line_kws={"color":"red"})
sns.scatterplot(x="ArkaKameraMP",y="OnKameraMP",data=df,hue="FiyatAraligi")
sns.jointplot(x=df["ArkaKameraMP"],y=df["OnKameraMP"],data=df,joint_kws={"s":5,"color":"red"},marginal_kws={"color":"red"})
df["4G"].unique()
df["4G"]=df["4G"].map({"Var":1,"Yok":0})
df["4G"].head()
sns.violinplot(x="FiyatAraligi",y="4G",data=df)
sns.violinplot(x="FiyatAraligi",y="DahiliBellek",data=df)
sns.boxplot(df["DahiliBellek"])
sns.distplot(df["DahiliBellek"],bins=20)
sns.kdeplot(df["DahiliBellek"],shade=True)
sns.violinplot(x="FiyatAraligi",y="Kalinlik",data=df)
sns.boxplot(df["Kalinlik"])
sns.distplot(df["Kalinlik"],bins=20)
sns.kdeplot(df["Kalinlik"],shade=True)
sns.violinplot(x="FiyatAraligi",y="Agirlik",data=df)
sns.boxplot(df["Agirlik"])
sns.distplot(df["Agirlik"],bins=20)
sns.kdeplot(df["Agirlik"],shade=True)
sns.violinplot(x="FiyatAraligi",y="CekirdekSayisi",data=df)
sns.boxplot(df["CekirdekSayisi"])
sns.distplot(df["CekirdekSayisi"],bins=20)
sns.kdeplot(df["CekirdekSayisi"],shade=True)
sns.violinplot(x="FiyatAraligi",y="CozunurlukYükseklik",data=df)
sns.boxplot(df["CozunurlukYükseklik"])
Q1=df["CozunurlukYükseklik"].quantile(0.25)

Q3=df["CozunurlukYükseklik"].quantile(0.75)

IQR=Q3-Q1

altsinir_cozunurlukyükseklik=Q1-1.5*IQR

ustsinir_cozunurlukyükseklik=Q3+1.5*IQR

print("AltSınır:{0}\nÜstSınır:{1}".format(altsinir_cozunurlukyükseklik,ustsinir_cozunurlukyükseklik))

dfCozunurlukYükseklik=df["CozunurlukYükseklik"]

aykiri_CozunurlukYükseklik=((dfCozunurlukYükseklik<0)|(dfCozunurlukYükseklik>1944.0))

dfCozunurlukYükseklik[aykiri_CozunurlukYükseklik]=1944.0
sns.boxplot(df["CozunurlukYükseklik"])
sns.distplot(df["CozunurlukYükseklik"],bins=20)
sns.kdeplot(df["CozunurlukYükseklik"],shade=True)
sns.violinplot(x="FiyatAraligi",y="CozunurlukGenislik",data=df)
sns.boxplot(df["CozunurlukGenislik"])
sns.distplot(df["CozunurlukGenislik"],bins=20)
sns.kdeplot(df["CozunurlukGenislik"],shade=True)
sns.scatterplot(x = df["CozunurlukYükseklik"], y = df["CozunurlukGenislik"])
sns.jointplot(x = df["CozunurlukYükseklik"], y = df["CozunurlukGenislik"])
#to do
sns.lmplot(x="CozunurlukYükseklik",y="CozunurlukGenislik",data=df,col="FiyatAraligi",scatter_kws={"color":"green"},line_kws={"color":"red"})
sns.scatterplot(x="CozunurlukYükseklik",y="CozunurlukGenislik",data=df,hue="FiyatAraligi")
sns.jointplot(x=df["CozunurlukYükseklik"],y=df["CozunurlukGenislik"],data=df,joint_kws={"s":5,"color":"red"},marginal_kws={"color":"red"})
sns.violinplot(x="FiyatAraligi",y="RAM",data=df)
sns.boxplot(df["RAM"])
sns.distplot(df["RAM"],bins=20)
sns.kdeplot(df["RAM"],shade=True)
sns.violinplot(x="FiyatAraligi",y="BataryaOmru",data=df)
sns.boxplot(df["BataryaOmru"])
sns.distplot(df["BataryaOmru"],bins=20)
sns.kdeplot(df["BataryaOmru"],shade=True)
df["3G"].unique()
df["3G"]=df["3G"].map({"Var":1,"Yok":0})
df["3G"].head()
sns.violinplot(x="FiyatAraligi",y="3G",data=df)
df["Dokunmatik"].unique()
df["Dokunmatik"]=df["Dokunmatik"].map({"Var":1,"Yok":0})
df["Dokunmatik"].head()
sns.violinplot(x="FiyatAraligi",y="Dokunmatik",data=df)
df["WiFi"].unique()
df["WiFi"]=df["WiFi"].map({"Var":1,"Yok":0})
df["WiFi"].head()
sns.violinplot(x="FiyatAraligi",y="WiFi",data=df)
df["Renk"].unique()
df["Renk"].value_counts()
df.Renk.value_counts().plot(kind = 'barh')
df["FiyatAraligi"].unique()
df["FiyatAraligi"]=df["FiyatAraligi"].map({"Çok Ucuz":0,"Ucuz":1,"Normal":2,"Pahalı":3})
df2=df.copy()
df2.head()
df2=pd.get_dummies(df2,columns=["Renk"],prefix=["Renk"])
df2.corr()
df.drop("Renk",axis=1,inplace=True)
df.head()
df.describe().T
df.info()
y=df["FiyatAraligi"]

X=df.drop(["FiyatAraligi"],axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
X_train
X_test
y_train
y_test
n_neighbors=[]
for i in range(2,16):

    knn_model=KNeighborsClassifier(n_neighbors=i).fit(X_train,y_train)

    y_pred= knn_model.predict(X_test)

    n_neighbors.append("n:{0} accuracy_score:{1} recall_score:{2} precision_score:{3} f1_score_score:{4} ".format(i,accuracy_score(y_test, y_pred),recall_score(y_test, y_pred,average="macro"),precision_score(y_test, y_pred,average="macro"),f1_score(y_test, y_pred,average="macro")))

n_neighbors
dataframe_neighbors=[]

dataframe_neighbors_accuracy=[]

dataframe_neighbors_recall=[]

dataframe_neighbors_precision=[]

dataframe_neighbors_f1=[]

for i in range(0,14):

        dataframe_neighbors.append(n_neighbors[i].split(" ")[0])

        dataframe_neighbors_accuracy.append(n_neighbors[i].split(" ")[1].split(":")[1])

        dataframe_neighbors_recall.append(n_neighbors[i].split(" ")[2].split(":")[1])

        dataframe_neighbors_precision.append(n_neighbors[i].split(" ")[3].split(":")[1])

        dataframe_neighbors_f1.append(n_neighbors[i].split(" ")[4].split(":")[1])

        
neighbors_comparison=pd.DataFrame({

    "n_neighbors":dataframe_neighbors,

    "Accuracy":dataframe_neighbors_accuracy,

    "Recall":dataframe_neighbors_recall,

    "Precision":dataframe_neighbors_precision,

    "F1-Score":dataframe_neighbors_f1},

    columns=["n_neighbors","Accuracy","Recall","Precision","F1-Score"])
neighbors_comparison
sns.barplot(y="Accuracy",x="n_neighbors",data=neighbors_comparison)
knn=KNeighborsClassifier(n_neighbors=11).fit(X_train,y_train)
y_pred_knn=knn.predict(X_test)
y_pred_knn
tahmin_gercek_knn=pd.DataFrame(y_pred_knn)

tahmin_gercek_knn["y_test"]=y_test.values

tahmin_gercek_knn.columns=["Tahmin","Gerçek"]

tahmin_gercek_knn
def tahmin_gercek(data):

    sns.lmplot(x = "Gerçek", y = "Tahmin", data =data)
tahmin_gercek(tahmin_gercek_knn)
print(classification_report(y_test,y_pred))
confusion_matrix(y_test,y_pred)
cart = DecisionTreeClassifier()
cart_params = {"max_depth": range(1,10),

              "min_samples_split": [1,2,3,4,5,10,20,25]}
cart_cv_model = GridSearchCV(cart, cart_params, cv = 10, n_jobs = -1, verbose =2).fit(X_train, y_train)
cart_cv_model.best_params_
cart_gini = DecisionTreeClassifier(max_depth = 9, min_samples_split = 10,criterion="gini").fit(X_train, y_train)
y_pred_gini=cart_gini.predict(X_test)
y_pred_gini
tahmin_gercek_gini=pd.DataFrame(y_pred_gini)

tahmin_gercek_gini["y_test"]=y_test.values

tahmin_gercek_gini.columns=["Tahmin","Gerçek"]

tahmin_gercek_gini
tahmin_gercek(tahmin_gercek_gini)
print(classification_report(y_test,y_pred_gini))
confusion_matrix(y_test,y_pred_gini)
cart_entropy=DecisionTreeClassifier(max_depth = 9, min_samples_split = 10,criterion="entropy").fit(X_train, y_train)
y_pred_entropy=cart_entropy.predict(X_test)
tahmin_gercek_entropy=pd.DataFrame(y_pred_entropy)

tahmin_gercek_entropy["y_test"]=y_test.values

tahmin_gercek_entropy.columns=["Tahmin","Gerçek"]

tahmin_gercek_entropy
tahmin_gercek(tahmin_gercek_entropy)
print(classification_report(y_test,y_pred_entropy))
confusion_matrix(y_test,y_pred_entropy)
accuracy_gini=accuracy_score(y_test,y_pred_gini)

accuracy_entropy=accuracy_score(y_test,y_pred_entropy)

recall_gini=recall_score(y_test, y_pred_gini,average="macro")

recall_entropy=recall_score(y_test, y_pred_entropy,average="macro")

precision_gini=precision_score(y_test, y_pred_gini,average="macro")

precision_entropy=precision_score(y_test, y_pred_entropy,average="macro")

f1_gini=f1_score(y_test, y_pred_gini,average="macro")

f1_entropy=f1_score(y_test, y_pred_entropy,average="macro")
criterion_comparision=pd.DataFrame({

    "criterion":['gini','entropy'],

    "Accuracy":[accuracy_gini,accuracy_entropy],

    "Recall":[recall_gini,recall_entropy],

    "Precision":[precision_gini,precision_entropy],

    "F1-Score":[f1_gini,f1_entropy],

    

},columns=["criterion","Accuracy","Recall","Precision","F1-Score"])

criterion_comparision
gnb=GaussianNB().fit(X_train,y_train)
y_pred_gnb=gnb.predict(X_test)
gnb_accuracy=accuracy_score(y_test, y_pred_gnb)

gnb_recall=recall_score(y_test, y_pred_gnb,average="macro")

gnb_precision=precision_score(y_test,y_pred_gnb,average="macro")

gnb_f1_score=f1_score(y_test,y_pred_gnb,average="macro")
tahmin_gercek_gnb=pd.DataFrame(y_pred_gnb)

tahmin_gercek_gnb["y_test"]=y_test.values

tahmin_gercek_gnb.columns=["Tahmin","Gerçek"]

tahmin_gercek_gnb
tahmin_gercek(tahmin_gercek_gnb)
confusion_matrix(y_test,y_pred_gnb)
print(classification_report(y_test,y_pred))
model_karsilastirma=pd.DataFrame({

    "Model":["K-Nearest","DecisionTree","NaiveBayes"],

    "Accuracy":[n_neighbors[9].split(" ")[1].split(":")[1],accuracy_entropy,gnb_accuracy],

    "Recall":[n_neighbors[9].split(" ")[2].split(":")[1],recall_entropy,gnb_recall],

    "Precision":[n_neighbors[9].split(" ")[3].split(":")[1],precision_entropy,gnb_precision],

    "F1-Score":[n_neighbors[9].split(" ")[4].split(":")[1],f1_entropy,gnb_f1_score],

},columns=["Model","Accuracy","Recall","Precision","F1-Score"])

model_karsilastirma