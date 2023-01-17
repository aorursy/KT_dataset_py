import numpy as np 

import pandas as pd

import seaborn as sns

import missingno as msno

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve,classification_report,f1_score, accuracy_score, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve,auc

from sklearn.model_selection import cross_val_predict

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.tools as tls

import statsmodels.api as sm

import warnings

warnings.filterwarnings('ignore') 
df=pd.read_csv("../input/tablet/tablet.csv")

df.head()
df.shape
df.ndim
df.size
df.describe().T
df["FiyatAraligi"].unique()
df["FiyatAraligi"].value_counts()
data = go.Bar( x = df['FiyatAraligi'].value_counts()

              , y = ['Normal','Pahalı','Ucuz','Çok Ucuz' ], text=df['FiyatAraligi'].value_counts(),

                    orientation = 'h',textfont=dict(size=15),

                    textposition = 'auto')





fig = go.Figure(data = data, layout={"title":'Ürün Değerleri'})

py.iplot(fig)

df.info()
df.corr()
df.cov()
corr=df.corr()



sns.set(font_scale=1.15)

plt.figure(figsize=(14, 10))



sns.heatmap(corr, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='YlGnBu',linecolor="black")

plt.title('Değişkenlerin Birbiri ile Korelasyonu');
df.isnull().sum()
df.isnull().sum().sum()
msno.bar(df)
msno.matrix(df)
msno.heatmap(df)
df[df.isnull().any(axis=1)]
df.loc[(df['ArkaKameraMP'] == 0 ) & (df['OnKameraMP'].isnull()), 'OnKameraMP'] = 0.0
df_eksik=df[df.isnull().any(axis=1)]

df_eksik
df.groupby("FiyatAraligi")["RAM"].mean()
df["RAM"].mean()
df["RAM"].fillna(df.groupby("FiyatAraligi")["RAM"].transform("mean"),inplace=True)
df[df.isnull().any(axis=1)]
df.groupby("FiyatAraligi")["OnKameraMP"].mean()
df["OnKameraMP"].fillna(4,inplace=True)
df.isnull().sum().sum()
df.head()
def violin_ciz(x,y,data):

    sns.set_style("whitegrid")

    sns.violinplot(x=x,y=y,data=data)
df_cokucuz=df[(df['FiyatAraligi'] != "Çok Ucuz")]

df_ucuz=df[(df['FiyatAraligi'] != "Ucuz")]

df_normal=df[(df['FiyatAraligi'] != "Normal")]

df_pahali=df[(df['FiyatAraligi'] != "Pahalı")]



def graph_func(variable):

    tmp_cokucuz=df_cokucuz[variable]

    tmp_ucuz=df_ucuz[variable]

    tmp_normal=df_normal[variable]

    tmp_pahali=df_pahali[variable]

    

    hist_tmp=[tmp_cokucuz,tmp_ucuz,tmp_normal,tmp_pahali]

    layouts=["Çok Ucuz","Ucuz","Normal","Pahalı"]

    fig=ff.create_distplot(hist_tmp,layouts,show_hist=True,bin_size=0,curve_type="kde")

    py.iplot(fig)
violin_ciz("FiyatAraligi","BataryaGucu",df)
sns.boxplot(df["BataryaGucu"])
sns.distplot(df["BataryaGucu"],bins=16,color="purple")
graph_func("BataryaGucu")
violin_ciz("FiyatAraligi","MikroislemciHizi",df)
sns.boxplot(df["MikroislemciHizi"])
sns.distplot(df["MikroislemciHizi"],bins=16,color="purple")
graph_func("MikroislemciHizi")
violin_ciz("FiyatAraligi","OnKameraMP",df)
sns.boxplot(df["OnKameraMP"])
sns.distplot(df["OnKameraMP"],bins=16,color="purple")
graph_func("OnKameraMP")
violin_ciz("FiyatAraligi","DahiliBellek",df)
sns.boxplot(df["DahiliBellek"])
sns.distplot(df["DahiliBellek"],bins=16,color="purple")
graph_func("DahiliBellek")
violin_ciz("FiyatAraligi","Kalinlik",df)
sns.boxplot(df["Kalinlik"])
sns.distplot(df["Kalinlik"],bins=16,color="purple")
graph_func("Kalinlik")
violin_ciz("FiyatAraligi","Agirlik",df)
sns.boxplot(df["Agirlik"])
sns.distplot(df["Agirlik"],bins=16,color="purple")
graph_func("Agirlik")
violin_ciz("FiyatAraligi","CekirdekSayisi",df)
sns.boxplot(df["CekirdekSayisi"])
sns.distplot(df["CekirdekSayisi"],bins=16,color="purple")
graph_func("CekirdekSayisi")
violin_ciz("FiyatAraligi","ArkaKameraMP",df)
sns.boxplot(df["ArkaKameraMP"])
sns.distplot(df["ArkaKameraMP"],bins=16,color="purple")
graph_func("ArkaKameraMP")
violin_ciz("FiyatAraligi","CozunurlukYükseklik",df)
sns.boxplot(df["CozunurlukYükseklik"])
sns.distplot(df["CozunurlukYükseklik"],bins=16,color="purple")
graph_func("CozunurlukYükseklik")
violin_ciz("FiyatAraligi","CozunurlukGenislik",df)
sns.boxplot(df["CozunurlukGenislik"])
sns.distplot(df["CozunurlukGenislik"],bins=16,color="purple")
graph_func("CozunurlukGenislik")
violin_ciz("FiyatAraligi","RAM",df)
sns.boxplot(df["RAM"])
sns.distplot(df["RAM"],bins=16,color="purple")
graph_func("RAM")
violin_ciz("FiyatAraligi","BataryaOmru",df)
sns.boxplot(df["BataryaOmru"])
sns.distplot(df["BataryaOmru"],bins=16,color="purple")
graph_func("BataryaOmru")
sns.lmplot(x="OnKameraMP",y="ArkaKameraMP",data=df,scatter_kws={"color":"green"},line_kws={"color":"red"})
sns.scatterplot(x="OnKameraMP",y="ArkaKameraMP",data=df,hue="FiyatAraligi")
sns.jointplot(x=df["OnKameraMP"],y=df["ArkaKameraMP"],data=df,joint_kws={"s":5,"color":"red"},marginal_kws={"color":"red"})
sns.lmplot(x="CozunurlukYükseklik",y="CozunurlukGenislik",data=df,scatter_kws={"color":"green"},line_kws={"color":"red"})
sns.scatterplot(x="CozunurlukYükseklik",y="CozunurlukGenislik",data=df,hue="FiyatAraligi")
sns.jointplot(x=df["CozunurlukYükseklik"],y=df["CozunurlukGenislik"],data=df,joint_kws={"s":5,"color":"red"},marginal_kws={"color":"red"})
def aykiri_gozlem_incele(x):

        Q1=df[x].quantile(0.25)

        Q3=df[x].quantile(0.75)

        IQR=Q3-Q1

        alt_sinir=Q1-1.5*IQR

        ust_sinir=Q3+1.5*IQR

        return alt_sinir,ust_sinir
liste=['BataryaGucu','MikroislemciHizi','OnKameraMP','DahiliBellek','Kalinlik','Agirlik','CekirdekSayisi',

       'ArkaKameraMP','CozunurlukYükseklik','CozunurlukGenislik','RAM','BataryaOmru']

dictionary_columns=dict()



for i in liste:

    

    dictionary_columns.setdefault(i,aykiri_gozlem_incele(i))





dictionary_columns
altsinirOnKamera=-8.0

ustsinirOnKamera=16.0
df[(df["OnKameraMP"]<altsinirOnKamera)|(df["OnKameraMP"]>ustsinirOnKamera)]
df_onkamera=df["OnKameraMP"]

aykiri_onkamera=((df_onkamera<altsinirOnKamera)|(df_onkamera>ustsinirOnKamera))

df_onkamera[aykiri_onkamera]=ustsinirOnKamera
sns.boxplot(df["OnKameraMP"])
altsinirCozunurlukYukseklik=-714.0

ustsinirCozunurlukYukseklik=1944.0
df[(df["CozunurlukYükseklik"]<altsinirCozunurlukYukseklik)|(df["CozunurlukYükseklik"]>ustsinirCozunurlukYukseklik)]
df_cozunurlukyukseklik=df["CozunurlukYükseklik"]

aykiri_cozunurlukyukseklik=((df_cozunurlukyukseklik<altsinirCozunurlukYukseklik)|(df_cozunurlukyukseklik>ustsinirCozunurlukYukseklik))

df_cozunurlukyukseklik[aykiri_cozunurlukyukseklik]=ustsinirCozunurlukYukseklik
sns.boxplot(df["CozunurlukYükseklik"])
df.columns
feautures=['BataryaGucu',  'MikroislemciHizi','DahiliBellek', 'Kalinlik', 'Agirlik', 'CekirdekSayisi','CozunurlukYükseklik', 'CozunurlukGenislik', 'RAM',

       'BataryaOmru']

hatali_veriler=dict()

for i in feautures:

    hatali_veriler.setdefault(i,df[df[i]==0].index)
hatali_veriler
df[df["CozunurlukYükseklik"]==0]
df[df["CozunurlukGenislik"]==994]
df.loc[(df['CozunurlukYükseklik'] == 0 ) & (df['FiyatAraligi']=="Ucuz"), 'CozunurlukYükseklik'] = 363
df[df["CozunurlukGenislik"]==994]
df[df["CozunurlukGenislik"]==1987]
df.loc[(df['CozunurlukYükseklik'] == 0 ) & (df['FiyatAraligi']=="Pahalı"), 'CozunurlukYükseklik'] = 1197
df[df["CozunurlukGenislik"]==1987]
df_fiyatcokucuz=df["FiyatAraligi"]

aykiri_fiyatcokucuz=(df_fiyatcokucuz=="Çok Ucuz")

df_fiyatcokucuz[aykiri_fiyatcokucuz]=0
df_fiyatucuz=df["FiyatAraligi"]

aykiri_fiyatucuz=(df_fiyatucuz=="Ucuz")

df_fiyatucuz[aykiri_fiyatucuz]=1
df_fiyatnormal=df["FiyatAraligi"]

aykiri_fiyatnormal=(df_fiyatnormal=="Normal")

df_fiyatnormal[aykiri_fiyatnormal]=2
df_fiyatpahali=df["FiyatAraligi"]

aykiri_fiyatpahali=(df_fiyatpahali=="Pahalı")

df_fiyatpahali[aykiri_fiyatpahali]=3
df["FiyatAraligi"]=df['FiyatAraligi'].astype(str).astype(int)
df.info()
def strtonum(feature,string):

    df[feature]=np.where(df[feature].str.contains(string),1,0)

    
strtonum("Bluetooth","Var")

strtonum("CiftHat","Var")

strtonum("4G","Var")

strtonum("3G","Var")

strtonum("WiFi","Var")

strtonum("Dokunmatik","Var")
df.head()
df["Renk"].unique()
data = go.Bar( x = df['Renk'].value_counts()

              , y = ['Beyaz', 'Pembe', 'Mor', 'Turuncu', 'Gri', 'Sarı', 'Mavi',

       'Turkuaz', 'Kahverengi', 'Yeşil', 'Kırmızı', 'Siyah'], text=df['Renk'].value_counts(),

                    orientation = 'h',textfont=dict(size=15),

                    textposition = 'auto')





fig = go.Figure(data = data, layout={"title":'Ürün Renkleri'})

py.iplot(fig)

df.corr()
df.drop(columns=["Renk"],inplace=True)
df.head()
df.info()
df.describe().T
df.corr()
corr=df.corr()



sns.set(font_scale=1.15)

plt.figure(figsize=(14, 10))



sns.heatmap(corr, vmax=.8, linewidths=0.2,

            square=True,annot=True,cmap='YlGnBu',linecolor="black")

plt.title('Değişkenlerin Birbiri ile Korelasyonu');
df["FiyatAraligi"].value_counts()
y=df["FiyatAraligi"]

X=df.drop(["FiyatAraligi"],axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
X_train
y_train
y_test
X_test
knn=KNeighborsClassifier()
accuracy_score_knn=dict()

recall_score_knn=dict()

precision_score_knn=dict()

f1_score_knn=dict()

for i in range(2,16):

    knn_tuned=KNeighborsClassifier(n_neighbors = i).fit(X_train, y_train)

    y_pred1= knn_tuned.predict(X_test)

    accuracy_score_knn.setdefault(i,accuracy_score(y_test, y_pred1))

    recall_score_knn.setdefault(i,recall_score(y_test, y_pred1,average="macro")) 

    precision_score_knn.setdefault(i,precision_score(y_test, y_pred1,average="macro")) 

    f1_score_knn.setdefault(i,f1_score(y_test, y_pred1,average="macro")) 



model_comparison_knn=pd.DataFrame({

    "n_neighbors":["k=2","k=3","k=4","k=5","k=6","k=7","k=8","k=9","k=10","k=11","k=12","k=13","k=14","k=15"],

    "Accuracy":[accuracy_score_knn.get(2),accuracy_score_knn.get(3),accuracy_score_knn.get(4),accuracy_score_knn.get(5),accuracy_score_knn.get(6),accuracy_score_knn.get(7),accuracy_score_knn.get(8),accuracy_score_knn.get(9),accuracy_score_knn.get(10),accuracy_score_knn.get(11),accuracy_score_knn.get(12),accuracy_score_knn.get(13),accuracy_score_knn.get(14),accuracy_score_knn.get(15)],

    "Recall":[recall_score_knn.get(2),recall_score_knn.get(3),recall_score_knn.get(4),recall_score_knn.get(5),recall_score_knn.get(6),recall_score_knn.get(7),recall_score_knn.get(8),recall_score_knn.get(9),recall_score_knn.get(10),recall_score_knn.get(11),recall_score_knn.get(12),recall_score_knn.get(13),recall_score_knn.get(14),recall_score_knn.get(15)],

    "Precision":[precision_score_knn.get(2),precision_score_knn.get(3),precision_score_knn.get(4),precision_score_knn.get(5),precision_score_knn.get(6),precision_score_knn.get(7),precision_score_knn.get(8),precision_score_knn.get(9),precision_score_knn.get(10),precision_score_knn.get(11),precision_score_knn.get(12),precision_score_knn.get(13),precision_score_knn.get(14),precision_score_knn.get(15)],

    "F1-Score":[f1_score_knn.get(2),f1_score_knn.get(3),f1_score_knn.get(4),f1_score_knn.get(5),f1_score_knn.get(6),f1_score_knn.get(7),f1_score_knn.get(8),f1_score_knn.get(9),f1_score_knn.get(10),f1_score_knn.get(11),f1_score_knn.get(12),f1_score_knn.get(13),f1_score_knn.get(14),f1_score_knn.get(15)]

    

},columns=["n_neighbors","Accuracy","Recall","Precision","F1-Score"])
model_comparison_knn
knn_tuned=KNeighborsClassifier(n_neighbors=11).fit(X_train,y_train)
y_pred = knn_tuned.predict(X_test)
dictionary_knn_tuned=dict()
dictionary_knn_tuned.setdefault("Accuracy",accuracy_score(y_test, y_pred))

dictionary_knn_tuned.setdefault("Recall",recall_score(y_test, y_pred,average="macro"))

dictionary_knn_tuned.setdefault("Precision",precision_score(y_test,y_pred,average="macro"))

dictionary_knn_tuned.setdefault("F1_Score",f1_score(y_test,y_pred,average="macro"))

dictionary_knn_tuned
print(classification_report(y_test,y_pred))
confusion_matrix(y_test,y_pred)
sns.scatterplot(x="n_neighbors",y="Accuracy",data=model_comparison_knn)
plt.figure(figsize=(12,6))



plt.plot(model_comparison_knn["n_neighbors"],model_comparison_knn["Accuracy"])



plt.show()
cart = DecisionTreeClassifier()
cart_params = {"max_depth": [1,3,5,8,10],

              "min_samples_split": [2,3,5,10,20,50]}
cart_cv_model = GridSearchCV(cart, cart_params, cv = 10, n_jobs = -1, verbose =2).fit(X_train, y_train)
cart_cv_model.best_params_
cart_tuned = DecisionTreeClassifier(max_depth = 8, min_samples_split = 2).fit(X_train, y_train)
y_pred_cart = cart_tuned.predict(X_test)
dictionary_cart_tuned=dict()
dictionary_cart_tuned.setdefault("Accuracy",accuracy_score(y_test, y_pred_cart))

dictionary_cart_tuned.setdefault("Recall",recall_score(y_test, y_pred_cart,average="macro"))

dictionary_cart_tuned.setdefault("Precision",precision_score(y_test,y_pred_cart,average="macro"))

dictionary_cart_tuned.setdefault("F1_Score",f1_score(y_test,y_pred_cart,average="macro"))

dictionary_cart_tuned
print(classification_report(y_test,y_pred_cart))
confusion_matrix(y_test,y_pred_cart)
accuracy_score_cart=dict()

recall_score_cart=dict()

precision_score_cart=dict()

f1_score_cart=dict()

max_depth= [1,3,5,8,10]

for i in max_depth:

    cart_tuned=DecisionTreeClassifier(max_depth = i,min_samples_split = 2).fit(X_train, y_train)

    y_pred1_cart= cart_tuned.predict(X_test)

    accuracy_score_cart.setdefault(i,accuracy_score(y_test, y_pred1_cart))

    recall_score_cart.setdefault(i,recall_score(y_test, y_pred1_cart,average="macro")) 

    precision_score_cart.setdefault(i,precision_score(y_test, y_pred1_cart,average="macro")) 

    f1_score_cart.setdefault(i,f1_score(y_test, y_pred1_cart,average="macro")) 



model_comparison_cart=pd.DataFrame({

    "max_depth":["max_depth=1","max_depth=3","max_depth=5","max_depth=8","max_depth=10"],

    "Accuracy":[accuracy_score_cart.get(1),accuracy_score_cart.get(3),accuracy_score_cart.get(5),accuracy_score_cart.get(8),accuracy_score_cart.get(10)],

    "Recall":[recall_score_cart.get(1),recall_score_cart.get(3),recall_score_cart.get(5),recall_score_cart.get(8),recall_score_cart.get(10)],

    "Precision":[precision_score_cart.get(1),precision_score_cart.get(3),precision_score_cart.get(5),precision_score_cart.get(8),precision_score_cart.get(10)],

    "F1-Score":[f1_score_cart.get(1),f1_score_cart.get(3),f1_score_cart.get(5),f1_score_cart.get(8),f1_score_cart.get(10)]

    

},columns=["max_depth","Accuracy","Recall","Precision","F1-Score"])
model_comparison_cart
accuracy_score_cart_split=dict()

recall_score_cart_split=dict()

precision_score_cart_split=dict()

f1_score_cart_split=dict()

min_samples_split= [2,3,5,10,20,50]

for i in min_samples_split:

    cart_tuned_split=DecisionTreeClassifier(max_depth = 8,min_samples_split=i).fit(X_train, y_train)

    y_pred1_cart_split= cart_tuned_split.predict(X_test)

    accuracy_score_cart_split.setdefault(i,accuracy_score(y_test, y_pred1_cart_split))

    recall_score_cart_split.setdefault(i,recall_score(y_test, y_pred1_cart_split,average="macro")) 

    precision_score_cart_split.setdefault(i,precision_score(y_test, y_pred1_cart_split,average="macro")) 

    f1_score_cart_split.setdefault(i,f1_score(y_test, y_pred1_cart_split,average="macro")) 
model_comparison_cart_split=pd.DataFrame({

    "min_samples_split":["min_samples_split=2","min_samples_split=3","min_samples_split=5","min_samples_split=10","min_samples_split=20","min_samples_split=50"],

    "Accuracy":[accuracy_score_cart_split.get(2),accuracy_score_cart_split.get(3),accuracy_score_cart_split.get(5),accuracy_score_cart_split.get(10),accuracy_score_cart_split.get(20),accuracy_score_cart_split.get(50)],

    "Recall":[recall_score_cart_split.get(2),recall_score_cart_split.get(3),recall_score_cart_split.get(5),recall_score_cart_split.get(10),recall_score_cart_split.get(20),recall_score_cart_split.get(50)],

    "Precision":[precision_score_cart_split.get(2),precision_score_cart_split.get(3),precision_score_cart_split.get(5),precision_score_cart_split.get(10),precision_score_cart_split.get(20),precision_score_cart_split.get(50)],

    "F1-Score":[f1_score_cart_split.get(2),f1_score_cart_split.get(3),f1_score_cart_split.get(5),f1_score_cart_split.get(10),f1_score_cart_split.get(20),f1_score_cart_split.get(50)]

    

},columns=["min_samples_split","Accuracy","Recall","Precision","F1-Score"])
model_comparison_cart_split
cart_entropy = DecisionTreeClassifier(max_depth = 8, min_samples_split = 2,criterion="entropy").fit(X_train, y_train)

y_entropy_pred = cart_entropy.predict(X_test)

cart_entropy_accuracy=accuracy_score(y_test, y_entropy_pred)

cart_entropy_recall=recall_score(y_test, y_entropy_pred,average="macro")

cart_entropy_precision=precision_score(y_test,y_entropy_pred,average="macro")

cart_entropy_f1_score=f1_score(y_test,y_entropy_pred,average="macro")
model_comparison_cart_gini=pd.DataFrame({

    "criterion":["criterion='gini'","criterion='entropy'"],

    "Accuracy":[dictionary_cart_tuned.get("Accuracy"),cart_entropy_accuracy],

    "Recall":[dictionary_cart_tuned.get("Recall"),cart_entropy_recall],

    "Precision":[dictionary_cart_tuned.get("Precision"),cart_entropy_precision],

    "F1-Score":[dictionary_cart_tuned.get("F1_Score"),cart_entropy_f1_score],

    

},columns=["criterion","Accuracy","Recall","Precision","F1-Score"])

model_comparison_cart_gini
gnb_model=GaussianNB().fit(X_train,y_train)
gnb_model
y_pred=gnb_model.predict(X_test)
y_pred
nb_accuracy=accuracy_score(y_test, y_pred)

nb_recall=recall_score(y_test, y_pred,average="macro")

nb_precision=precision_score(y_test,y_pred,average="macro")

nb_f1_score=f1_score(y_test,y_pred,average="macro")
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
model_comparison=pd.DataFrame({

    "Model":["K-Nearest","DecisionTree","NaiveBayes"],

    "Accuracy":[dictionary_knn_tuned.get("Accuracy"),dictionary_cart_tuned.get("Accuracy"),nb_accuracy],

    "Recall":[dictionary_knn_tuned.get("Recall"),dictionary_cart_tuned.get("Recall"),nb_recall],

    "Precision":[dictionary_knn_tuned.get("Precision"),dictionary_cart_tuned.get("Precision"),nb_precision],

    "F1-Score":[dictionary_knn_tuned.get("F1_Score"),dictionary_cart_tuned.get("F1_Score"),nb_f1_score],

},columns=["Model","Accuracy","Recall","Precision","F1-Score"])

model_comparison.sort_values(by='Accuracy', ascending=False)