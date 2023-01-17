import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn import preprocessing
import missingno as msno
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from sklearn.model_selection import  GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from warnings import filterwarnings
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.metrics import confusion_matrix as cm

filterwarnings('ignore')

df=pd.read_csv("../input/tabletpc-priceclassification/tablet.csv")
df.head()
df.sample(5)
df.tail()
df.info()
df.columns
df.shape
df.describe()
df.isnull().sum().sum()
df.isnull().sum()
msno.matrix(df,figsize=(20, 10));
msno.heatmap(df, figsize= (15,8));
df[df["RAM"].isnull()]
df[df["FiyatAraligi"]=="Pahalı"].describe()["RAM"]
RamOrtalama=df[df["FiyatAraligi"]=="Pahalı"].mean()["RAM"]
df['RAM'].fillna(RamOrtalama, inplace = True)
df[df["OnKameraMP"].isnull()]
df[df["FiyatAraligi"]=="Çok Ucuz"].describe()["OnKameraMP"]
OnKameraMPortalama=df[df["FiyatAraligi"]=="Çok Ucuz"].mean()["OnKameraMP"]
df['OnKameraMP'].fillna(OnKameraMPortalama, inplace = True)
df.isnull().sum().sum()
df.isnull().sum()
df.head()
df.info()
df["Bluetooth"].value_counts()
df["CiftHat"].value_counts()
df["4G"].value_counts()
df["3G"].value_counts()
df["Dokunmatik"].value_counts()
df["WiFi"].value_counts()
df["FiyatAraligi"].value_counts()
df.replace("Yok",0,inplace=True)
df.replace("Var",1,inplace=True)
df.head()
df["FiyatAraligi"]=df["FiyatAraligi"].replace("Çok Ucuz",0)
df["FiyatAraligi"]=df["FiyatAraligi"].replace("Ucuz",1)
df["FiyatAraligi"]=df["FiyatAraligi"].replace("Normal",2)
df["FiyatAraligi"]=df["FiyatAraligi"].replace("Pahalı",3)
df.head()
df.info()
df["Renk"].value_counts()
trace1=go.Bar(
    x=df[df.FiyatAraligi==0].Renk.value_counts().index,
    y=df[df.FiyatAraligi==0].Renk.value_counts().values,
    name="Çok Ucuz",
    marker=dict(color="rgba(200,0,0,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace2=go.Bar(
    x=df[df.FiyatAraligi==1].Renk.value_counts().index,
    y=df[df.FiyatAraligi==1].Renk.value_counts().values,
    name="Ucuz",
    marker=dict(color="rgba(0,200,0,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace3=go.Bar(
    x=df[df.FiyatAraligi==2].Renk.value_counts().index,
    y=df[df.FiyatAraligi==2].Renk.value_counts().values,
    name="Normal",
    marker=dict(color="rgba(0,0,200,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace4=go.Bar(
    x=df[df.FiyatAraligi==3].Renk.value_counts().index,
    y=df[df.FiyatAraligi==3].Renk.value_counts().values,
    name="Pahalı",
    marker=dict(color="rgba(50,20,100,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )


data=[trace1,trace2,trace3,trace4]
layout= go.Layout(barmode="group")

fig=go.Figure(data=data,layout=layout)
iplot(fig)
df.describe().T
df.corr()
f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(df.corr(),annot=True,linewidths=5,fmt='.0%',ax=ax)
plt.show()
fig1, ax1 = plt.subplots(1,2,figsize=(8,8))
sns.countplot(df['FiyatAraligi'],ax=ax1[0])
labels = 'ÇOK UCUZ', 'UCUZ', 'NORMAL','PAHALI'
df.FiyatAraligi.value_counts().plot.pie(labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
plt.show()
plt.figure(figsize=(18,5))
sns.lmplot(x="ArkaKameraMP", y="OnKameraMP", data=df)
plt.title('ArkaKameraMP- OnKameraMP')
plt.show()
plt.figure(figsize=(18,5))
sns.scatterplot(x = "ArkaKameraMP", y = "OnKameraMP", hue="FiyatAraligi", data = df);
plt.title('ArkaKameraMP- OnKameraMP"nın FiyatAraligi Üzerinde Etkisi' )
plt.show()
plt.figure(figsize=(20,5))
sns.scatterplot(x = "CozunurlukGenislik", y = "CozunurlukYükseklik", data = df);
plt.title('CozunurlukGenislik- CozunurlukYükseklik')
plt.show()
plt.figure(figsize=(18,5))
sns.violinplot(x="FiyatAraligi", y="RAM", data=df)
plt.title('FiyatAraligi- RAM')
plt.show()
plt.figure(figsize=(18,5))
sns.violinplot(x="FiyatAraligi", y="BataryaGucu", data=df)
plt.title('FiyatAraligi- BataryaGucu')
plt.show()
plt.figure(figsize=(18,5))
sns.violinplot(x = "FiyatAraligi", y = "CozunurlukYükseklik", data = df);
plt.title('FiyatAraligi- CozunurlukYükseklik')
plt.show()
plt.figure(figsize=(18,5))
sns.violinplot(x = "FiyatAraligi", y = "CozunurlukGenislik", data = df);
plt.title('FiyatAraligi- CozunurlukGenislik')
plt.show()
trace1=go.Bar(
    x=df[df.FiyatAraligi==0].CiftHat.value_counts().index,
    y=df[df.FiyatAraligi==0].CiftHat.value_counts().values,
    name="Çok Ucuz",
    marker=dict(color="rgba(200,0,0,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace2=go.Bar(
    x=df[df.FiyatAraligi==1].CiftHat.value_counts().index,
    y=df[df.FiyatAraligi==1].CiftHat.value_counts().values,
    name="Ucuz",
    marker=dict(color="rgba(0,200,0,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace3=go.Bar(
    x=df[df.FiyatAraligi==2].CiftHat.value_counts().index,
    y=df[df.FiyatAraligi==2].CiftHat.value_counts().values,
    name="Normal",
    marker=dict(color="rgba(0,0,200,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace4=go.Bar(
    x=df[df.FiyatAraligi==3].CiftHat.value_counts().index,
    y=df[df.FiyatAraligi==3].CiftHat.value_counts().values,
    name="Pahalı",
    marker=dict(color="rgba(50,20,100,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )


data=[trace1,trace2,trace3,trace4]
layout= go.Layout(barmode="group")

fig=go.Figure(data=data,layout=layout)
iplot(fig)
trace1=go.Bar(
    x=df[df.FiyatAraligi==0].Dokunmatik.value_counts().index,
    y=df[df.FiyatAraligi==0].Dokunmatik.value_counts().values,
    name="Çok Ucuz",
    marker=dict(color="rgba(200,0,0,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace2=go.Bar(
    x=df[df.FiyatAraligi==1].Dokunmatik.value_counts().index,
    y=df[df.FiyatAraligi==1].Dokunmatik.value_counts().values,
    name="Ucuz",
    marker=dict(color="rgba(0,200,0,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace3=go.Bar(
    x=df[df.FiyatAraligi==2].Dokunmatik.value_counts().index,
    y=df[df.FiyatAraligi==2].Dokunmatik.value_counts().values,
    name="Normal",
    marker=dict(color="rgba(0,0,200,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace4=go.Bar(
    x=df[df.FiyatAraligi==3].Dokunmatik.value_counts().index,
    y=df[df.FiyatAraligi==3].Dokunmatik.value_counts().values,
    name="Pahalı",
    marker=dict(color="rgba(50,20,100,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )


data=[trace1,trace2,trace3,trace4]
layout= go.Layout(barmode="group")

fig=go.Figure(data=data,layout=layout)
iplot(fig)
trace1=go.Bar(
    x=df[df.FiyatAraligi==0].WiFi.value_counts().index,
    y=df[df.FiyatAraligi==0].WiFi.value_counts().values,
    name="Çok Ucuz",
    marker=dict(color="rgba(200,0,0,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace2=go.Bar(
    x=df[df.FiyatAraligi==1].WiFi.value_counts().index,
    y=df[df.FiyatAraligi==1].WiFi.value_counts().values,
    name="Ucuz",
    marker=dict(color="rgba(0,200,0,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace3=go.Bar(
    x=df[df.FiyatAraligi==2].WiFi.value_counts().index,
    y=df[df.FiyatAraligi==2].WiFi.value_counts().values,
    name="Normal",
    marker=dict(color="rgba(0,0,200,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace4=go.Bar(
    x=df[df.FiyatAraligi==3].WiFi.value_counts().index,
    y=df[df.FiyatAraligi==3].WiFi.value_counts().values,
    name="Pahalı",
    marker=dict(color="rgba(50,20,100,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )


data=[trace1,trace2,trace3,trace4]
layout= go.Layout(barmode="group")

fig=go.Figure(data=data,layout=layout)
iplot(fig)
trace1=go.Bar(
    x=df[df.FiyatAraligi==0].Bluetooth.value_counts().index,
    y=df[df.FiyatAraligi==0].Bluetooth.value_counts().values,
    name="Çok Ucuz",
    marker=dict(color="rgba(200,0,0,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace2=go.Bar(
    x=df[df.FiyatAraligi==1].Bluetooth.value_counts().index,
    y=df[df.FiyatAraligi==1].Bluetooth.value_counts().values,
    name="Ucuz",
    marker=dict(color="rgba(0,200,0,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace3=go.Bar(
    x=df[df.FiyatAraligi==2].Bluetooth.value_counts().index,
    y=df[df.FiyatAraligi==2].Bluetooth.value_counts().values,
    name="Normal",
    marker=dict(color="rgba(0,0,200,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace4=go.Bar(
    x=df[df.FiyatAraligi==3].Bluetooth.value_counts().index,
    y=df[df.FiyatAraligi==3].Bluetooth.value_counts().values,
    name="Pahalı",
    marker=dict(color="rgba(50,20,100,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )


data=[trace1,trace2,trace3,trace4]
layout= go.Layout(barmode="group")

fig=go.Figure(data=data,layout=layout)
iplot(fig)
df = df.rename(columns = {"3G":"G3","4G":"G4"})
trace1=go.Bar(
    x=df[df.FiyatAraligi==0].G3.value_counts().index,
    y=df[df.FiyatAraligi==0].G3.value_counts().values,
    name="Çok Ucuz",
    marker=dict(color="rgba(200,0,0,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace2=go.Bar(
    x=df[df.FiyatAraligi==1].G3.value_counts().index,
    y=df[df.FiyatAraligi==1].G3.value_counts().values,
    name="Ucuz",
    marker=dict(color="rgba(0,200,0,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace3=go.Bar(
    x=df[df.FiyatAraligi==2].G3.value_counts().index,
    y=df[df.FiyatAraligi==2].G3.value_counts().values,
    name="Normal",
    marker=dict(color="rgba(0,0,200,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace4=go.Bar(
    x=df[df.FiyatAraligi==3].G3.value_counts().index,
    y=df[df.FiyatAraligi==3].G3.value_counts().values,
    name="Pahalı",
    marker=dict(color="rgba(50,20,100,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )


data=[trace1,trace2,trace3,trace4]
layout= go.Layout(barmode="group")

fig=go.Figure(data=data,layout=layout)
iplot(fig)
trace1=go.Bar(
    x=df[df.FiyatAraligi==0].G4.value_counts().index,
    y=df[df.FiyatAraligi==0].G4.value_counts().values,
    name="Çok Ucuz",
    marker=dict(color="rgba(200,0,0,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace2=go.Bar(
    x=df[df.FiyatAraligi==1].G4.value_counts().index,
    y=df[df.FiyatAraligi==1].G4.value_counts().values,
    name="Ucuz",
    marker=dict(color="rgba(0,200,0,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace3=go.Bar(
    x=df[df.FiyatAraligi==2].G4.value_counts().index,
    y=df[df.FiyatAraligi==2].G4.value_counts().values,
    name="Normal",
    marker=dict(color="rgba(0,0,200,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace4=go.Bar(
    x=df[df.FiyatAraligi==3].G4.value_counts().index,
    y=df[df.FiyatAraligi==3].G4.value_counts().values,
    name="Pahalı",
    marker=dict(color="rgba(50,20,100,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )


data=[trace1,trace2,trace3,trace4]
layout= go.Layout(barmode="group")

fig=go.Figure(data=data,layout=layout)
iplot(fig)
trace1=go.Bar(
    x=df[df.FiyatAraligi==0].CekirdekSayisi.value_counts().index,
    y=df[df.FiyatAraligi==0].CekirdekSayisi.value_counts().values,
    name="Çok Ucuz",
    marker=dict(color="rgba(200,0,0,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace2=go.Bar(
    x=df[df.FiyatAraligi==1].CekirdekSayisi.value_counts().index,
    y=df[df.FiyatAraligi==1].CekirdekSayisi.value_counts().values,
    name="Ucuz",
    marker=dict(color="rgba(0,200,0,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace3=go.Bar(
    x=df[df.FiyatAraligi==2].CekirdekSayisi.value_counts().index,
    y=df[df.FiyatAraligi==2].CekirdekSayisi.value_counts().values,
    name="Normal",
    marker=dict(color="rgba(0,0,200,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )

trace4=go.Bar(
    x=df[df.FiyatAraligi==3].CekirdekSayisi.value_counts().index,
    y=df[df.FiyatAraligi==3].CekirdekSayisi.value_counts().values,
    name="Pahalı",
    marker=dict(color="rgba(50,20,100,0.5)",
               line=dict(color="rgba(0,0,0)",width=2)
               ),
    )


data=[trace1,trace2,trace3,trace4]
layout= go.Layout(barmode="group")

fig=go.Figure(data=data,layout=layout)
iplot(fig)
df=df.drop(["Renk"],axis=1)
df.head()
y=df["FiyatAraligi"]
y
X=df.drop(['FiyatAraligi'], axis=1)
X
from sklearn.model_selection import train_test_split
X_train, X_test ,y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=1)
tabletDT=DecisionTreeClassifier()
tabletDT_model=tabletDT.fit(X_train,y_train)
y_pred=tabletDT_model.predict(X_test)
accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
cm
tabletDT=DecisionTreeClassifier(criterion='entropy')
tabletDT_model=tabletDT.fit(X_train,y_train)
y_pred=tabletDT_model.predict(X_test)
accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
cm
cross_val_score(tabletDT_model,X,y, cv=10)
cross_val_score(tabletDT_model,X,y, cv=10).mean()
print(classification_report(y_test, y_pred))
ranking = tabletDT.feature_importances_
features = np.argsort(ranking)[::-1][:10]
columns = X.columns

plt.figure(figsize = (16, 9))
plt.title("Karar Ağacına Göre Özniteliklerin Önem Derecesi", y = 1.03, size = 18)
plt.bar(range(len(features)), ranking[features], color="lime", align="center")
plt.xticks(range(len(features)), columns[features], rotation=80)
plt.show()
tabletDT_grid={"max_depth": range(1,20),
            "min_samples_split" : range(2,50)}
dt = DecisionTreeClassifier()
tabletcv = GridSearchCV(dt, tabletDT_grid, cv = 10, n_jobs = -1, verbose = 2)
tabletcv_model=tabletcv.fit(X_train,y_train)
print("En iyi parametreler : " + str(tabletcv_model.best_params_))
print("En iyi skor : " + str(tabletcv_model.best_score_))
tabletdt = DecisionTreeClassifier(max_depth = 8, min_samples_split =5, criterion='entropy')
tablet_dt =tabletdt.fit(X_train, y_train)
y_pred=tablet_dt.predict(X_test)
accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
cm
knn_params = {"n_neighbors": np.arange(1,30)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, knn_params, cv = 3)
knn_cv.fit(X_train, y_train)
print("En iyi skor: " + str(knn_cv.best_score_))
print("En iyi parametreler: " + str(knn_cv.best_params_))
knn = KNeighborsClassifier(19)
knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
cm=confusion_matrix(y_test,y_pred)
cm
score_list = []

for each in range(1,16):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(X_train,y_train)
    score_list.append(knn2.score(X_test, y_test))
    
plt.plot(range(1,16),score_list)
plt.xlabel("k değerleri")
plt.ylabel("doğruluk skoru")
plt.show()
knn = KNeighborsClassifier(11)
knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
cm=confusion_matrix(y_test,y_pred)
cm
cross_val_score(knn_tuned, X_test, y_test, cv = 10)
cross_val_score(knn_tuned, X_test, y_test, cv = 10).mean()
print(classification_report(y_test, y_pred))
nb = GaussianNB()
nb_model = nb.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)
karmasiklik_matrisi
cross_val_score(nb,X_test,y_test,cv=10)
cross_val_score(nb,X_test,y_test,cv=10).mean()