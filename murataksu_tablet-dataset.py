import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
df=pd.read_csv("../input/tabletpc-priceclassification/tablet.csv")
df.head()
df.shape
df.info()
df.dtypes
df.isna().sum()
df.describe().T
df.cov()
corr=df.corr()
corr
sns.heatmap(df.corr())
sns.scatterplot(x = df["ArkaKameraMP"], y = df["OnKameraMP"])
sns.jointplot(x = df["CozunurlukYükseklik"], y = df["CozunurlukGenislik"])
df['FiyatAraligi'].unique()
df['FiyatAraligi'].nunique()
df['FiyatAraligi'].value_counts()
sns.distplot(df["BataryaGucu"], bins=40, color="purple");
sns.countplot(x="Bluetooth",data=df);
sns.barplot(x="FiyatAraligi",y="ArkaKameraMP",data=df);
sns.barplot(x="FiyatAraligi",y="RAM",data=df);
sns.barplot(x="FiyatAraligi",y="MikroislemciHizi",data=df);
sns.violinplot(y="MikroislemciHizi",data=df);
sns.violinplot(y="BataryaGucu",x="WiFi",data=df);
sns.jointplot(x=df["BataryaGucu"],y=df["BataryaOmru"],kind="kde",color="purple")
df["Bluetooth"].unique()
df["CiftHat"].unique()
df["MikroislemciHizi"].unique()
df["OnKameraMP"].unique()
df["4G"].unique()
df["Kalinlik"].unique()
df["CekirdekSayisi"].unique()
df["ArkaKameraMP"].unique()
df["3G"].unique()
df["Dokunmatik"].unique()
df["WiFi"].unique()
df["FiyatAraligi"].unique()
df["Renk"].unique()
msno.heatmap(df,figsize=(10,5))
df['WiFi'] = df['WiFi'].map({'Var': 1, 'Yok': 0})
df['Bluetooth'] = df['Bluetooth'].map({'Var': 1, 'Yok': 0})
df['CiftHat'] = df['CiftHat'].map({'Var': 1, 'Yok': 0})
df['4G'] = df['4G'].map({'Var': 1, 'Yok': 0})
df['3G'] = df['3G'].map({'Var': 1, 'Yok': 0})
df['Dokunmatik'] = df['Dokunmatik'].map({'Var': 1, 'Yok': 0})
df.head()
df["Renk"] = pd.Categorical(df["Renk"])
RenkDummies = pd.get_dummies(df["Renk"], prefix = "Renk")
df = pd.concat([df,RenkDummies],axis = 1)
df.head()
df.drop(["Renk","Renk_Kahverengi"],axis = 1, inplace = True)
df.head()
categories=pd.Categorical(df['FiyatAraligi'],categories=['Çok Ucuz','Ucuz','Normal','Pahalı'],ordered=True)
FiyatAraligi,unique=pd.factorize(categories,sort=True)
df['FiyatAraligi']=FiyatAraligi

df['FiyatAraligi']
df[df["RAM"].isnull()][["RAM","FiyatAraligi","CekirdekSayisi","MikroislemciHizi","DahiliBellek", "Bluetooth"]]
df[df["FiyatAraligi"] == 3]["RAM"].mean()
df[df["RAM"].isnull()][["RAM","FiyatAraligi","CekirdekSayisi","MikroislemciHizi","DahiliBellek", "Bluetooth"]]
df["RAM"].fillna(3449.3504098360654, inplace = True) 
df[df["RAM"].isnull()][["OnKameraMP","FiyatAraligi","CekirdekSayisi","MikroislemciHizi","DahiliBellek","Bluetooth"]]
df[df["FiyatAraligi"] == 0]["OnKameraMP"].mean() # Ucuz telefonların ortalaması
df[df["FiyatAraligi"] == 1]["OnKameraMP"].mean()
df[df["FiyatAraligi"] == 2]["OnKameraMP"].mean()
df[df["FiyatAraligi"] == 3]["OnKameraMP"].mean()
df["OnKameraMP"].mean() # Tüm telefonların ortalaması
df[df["FiyatAraligi"] == 3][["ArkaKameraMP","OnKameraMP"]].mean()
df[df["FiyatAraligi"] == 2][["ArkaKameraMP","OnKameraMP"]].mean()
df[df["FiyatAraligi"] == 1][["ArkaKameraMP","OnKameraMP"]].mean()
df[df["FiyatAraligi"] == 0][["ArkaKameraMP","OnKameraMP"]].mean() # Ucuz telefonların onkamera ve arka kamera ortalaması
df["OnKameraMP"].fillna(4.095123, inplace = True) 
df[df["OnKameraMP"].isnull()][["RAM","FiyatAraligi","ArkaKameraMP","MikroislemciHizi","DahiliBellek", "Bluetooth"]]
X = df.drop("FiyatAraligi",axis = 1)
Y = df["FiyatAraligi"]
X
Y
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 1/4)
X_train.count()
X_test.count()
Y_train.count()
Y_test.count()
Y_train
modeller = { 
    'Gercek Degerler': []
}

dfModel = pd.DataFrame(modeller, columns = ['Gercek Degerler'])
clf = GaussianNB()
clf.fit(X_train, Y_train)
y_pred_gaussian = clf.predict(X_test)

clf_decisiontree = DecisionTreeClassifier()

clf_train_decisiontree = clf_decisiontree.fit(X_train, Y_train)
y_pred_decisiontree = clf_train_decisiontree.predict(X_test)
KNNeigh = KNeighborsClassifier()
KNNeigh.fit(X_train, Y_train)
y_pred_KNNeigh = KNNeigh.predict(X_test)
print(classification_report(Y_test , y_pred_KNNeigh))

confusion_matrix(Y_test , y_pred_KNNeigh)
print(classification_report(Y_test, y_pred_gaussian))

confusion_matrix(Y_test, y_pred_gaussian)
print(classification_report(Y_test, y_pred_decisiontree))


confusion_matrix(Y_test, y_pred_decisiontree)
decision = classification_report(Y_test , y_pred_decisiontree,output_dict="true") # ilerde criterion='entropy' olan değişkenle karşılaştırmak için dict şeklinde aldık
dfModel["Gercek Degerler"] = df["FiyatAraligi"]
dfModel["GaussianModel"] = clf.predict(X)
dfModel["DecisionTreeModel"] = clf_decisiontree.predict(X)
dfModel["KNN_Model"] = neigh.predict(X)
dfModel.sample(10)
sns.lmplot(x = "Gercek Degerler", y = "GaussianModel", data = dfModel); 
sns.lmplot(x = "Gercek Degerler", y = "DecisionTreeModel", data = dfModel); 
sns.lmplot(x = "Gercek Degerler", y = "KNN_Model", data = dfModel); 
clf_decisiontree = DecisionTreeClassifier(criterion='entropy')
clf_train_decisiontree = clf_decisiontree.fit(X_train, Y_train)
y_pred_decision_entropy = clf_train_decisiontree.predict(X_test)
dfModel["DecisionTreeModel_Entropy"] = clf_decisiontree.predict(X)
dfModel.sample(10)[["Gercek Degerler","DecisionTreeModel","DecisionTreeModel_Entropy"]]
print(classification_report(Y_test , y_pred_decision_gini))
decision_yeni = classification_report(Y_test , y_pred_decision_gini,output_dict="true")
karsilastirma = { 
         'DecisionTreeClassifier': ['Gini','Entropy'],
         'Precision': [decision_yeni.get('weighted avg').get('precision') , decision_eski.get('weighted avg').get('precision')],
         'Recall': [decision_yeni.get('weighted avg').get('recall') , decision_eski.get('weighted avg').get('recall')],
         'F1-score': [decision_yeni.get('weighted avg').get('f1-score') , decision_eski.get('weighted avg').get('f1-score')],
         'Accuracy': [decision_yeni.get('accuracy') , decision_eski.get('accuracy')]
        }

df_Decision_Gini = pd.DataFrame(karsilastirma, columns = ['DecisionTreeClassifier','Precision','Recall','F1-score', 'Accuracy'])
df_Decision_Gini
knn_karsilastirma = {
         'Neighbors': [],
         'Precision': [],
         'Recall': [],
         'F1-score': [],
         'Accuracy': []
        }

df_knn_karsilastirma = pd.DataFrame(knn_karsilastirma, columns = ['Neighbors','Precision','Recall','F1-score', 'Accuracy'])
for x in range(2,16):
    neigh = KNeighborsClassifier(n_neighbors=x)
    neigh.fit(X_train, Y_train)
    y_pred =  neigh.predict(X_test)
    knn_report = classification_report(Y_test , y_pred,output_dict="true")
    df_knn_karsilastirma.loc[x-2] =[x,knn_report.get('weighted avg').get('precision'),knn_report.get('weighted avg').get('recall'),knn_report.get('weighted avg').get('f1-score'),knn_report.get('accuracy')]
    
df_knn_karsilastirma
sns.barplot(y="Accuracy",x="Neighbors",data=df_knn_karsilastirma)
sns.barplot(y="F1-score",x="Neighbors",data=df_knn_karsilastirma)
sns.barplot(y="Recall",x="Neighbors",data=df_knn_karsilastirma)
sns.barplot(y="Precision",x="Neighbors",data=df_knn_karsilastirma)