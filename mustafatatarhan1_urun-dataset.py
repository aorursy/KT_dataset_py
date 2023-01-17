import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


df=pd.read_csv("../input/tablet.csv")
df.head()
df.shape
df.info()
df['FiyatAraligi'].unique()
df['FiyatAraligi'].value_counts()
df.describe().T
corr=df.corr()
corr
sns.heatmap(df.corr())
sns.jointplot(x = df["ArkaKameraMP"], y = df["OnKameraMP"])
sns.countplot(x="Renk",data=df);
sns.distplot(df["MikroislemciHizi"], bins=40, color="purple");
sns.barplot(x="FiyatAraligi",y="CekirdekSayisi",data=df);
sns.barplot(x="FiyatAraligi",y="RAM",data=df);
sns.violinplot(y="BataryaGucu",data=df);
sns.jointplot(x=df["BataryaGucu"],y=df["BataryaOmru"],kind="kde",color="purple")
df["CiftHat"].unique()
df["MikroislemciHizi"].unique()
df["OnKameraMP"].unique()
df["Bluetooth"].unique()
df["Kalinlik"].unique()
df["4G"].unique()
df["CekirdekSayisi"].unique()
df["3G"].unique()
df["Dokunmatik"].unique()
df["WiFi"].unique()
df["ArkaKameraMP"].unique()
df["Renk"].unique()
df["FiyatAraligi"].unique()
df.isna().sum()
msno.heatmap(df,figsize=(10,5))
df['WiFi'] = df['WiFi'].map({'Var': 1, 'Yok': 0})
df['Bluetooth'] = df['Bluetooth'].map({'Var': 1, 'Yok': 0})
df['CiftHat'] = df['CiftHat'].map({'Var': 1, 'Yok': 0})
df['4G'] = df['4G'].map({'Var': 1, 'Yok': 0})
df['3G'] = df['3G'].map({'Var': 1, 'Yok': 0})
df['Dokunmatik'] = df['Dokunmatik'].map({'Var': 1, 'Yok': 0})
df["Renk"] = pd.Categorical(df["Renk"])
RenkDummies = pd.get_dummies(df["Renk"], prefix = "Renk")
df = pd.concat([df,RenkDummies],axis = 1)
df.head()
df.drop(["Renk","Renk_Turkuaz"],axis = 1, inplace = True)
df.head()
categories=pd.Categorical(df['FiyatAraligi'],categories=['Çok Ucuz','Ucuz','Normal','Pahalı'],ordered=True)
FiyatAraligi,unique=pd.factorize(categories,sort=True)
df['FiyatAraligi']=FiyatAraligi
df.head()

df[df["RAM"].isnull()][["RAM","FiyatAraligi"]]
df[df["FiyatAraligi"] == 3]["RAM"].mean()
df["RAM"].fillna(3449.3504098360654, inplace = True) 
df[df["OnKameraMP"].isnull()][["OnKameraMP","FiyatAraligi"]]
df[df["FiyatAraligi"] == 0]["OnKameraMP"].mean()
df["OnKameraMP"].fillna(4.092929292929293, inplace = True) 
X = df.drop("FiyatAraligi",axis = 1)
y = df["FiyatAraligi"]
X
y
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/4)
X_train.count()
X_test.count()
y_train.count()
y_test.count()
y_train
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred_gaussian = clf.predict(X_test)
df["Gaussian Tahmini"] = clf.predict(X)
df.sample(10)[["FiyatAraligi","Gaussian Tahmini"]]
confusion_matrix(y_test, y_pred_gaussian)
print(classification_report(y_test, y_pred_gaussian))

clf_decisiontree = DecisionTreeClassifier()

clf_train_decisiontree = clf_decisiontree.fit(X_train, y_train)
y_pred_decisiontree = clf_train_decisiontree.predict(X_test)
df["Decision Tahmini"] = clf_decisiontree.predict(X)
df.sample(10)[["FiyatAraligi","Decision Tahmini"]]
confusion_matrix(y_test, y_pred_decisiontree)
print(classification_report(y_test, y_pred_decisiontree))


KNN = KNeighborsClassifier()


KNN.fit(X_train, y_train)
y_pred_KNN = KNN.predict(X_test)
df["KNN Tahmini"] = KNN.predict(X)
df.sample(10)[["FiyatAraligi","KNN Tahmini"]]
confusion_matrix(y_test , y_pred_KNN)
print(classification_report(y_test , y_pred_KNN))

decision = classification_report(Y_test , y_pred_decisiontree,output_dict="true")
clf_decisionEntropy = DecisionTreeClassifier(criterion='entropy')
clf_train_decisionEntropy = clf_decisionEntropy.fit(X_train, y_train)
y_pred_decisionEntropy = clf_train_decisionEntropy.predict(X_test)
df["Decision Entropy Tahmini"] = clf_decisionEntropy.predict(X)
df.sample(10)[["FiyatAraligi","Decision Entropy Tahmini"]]
DecisionEntropyTahmin = np.asarray(df['Decision Entropy Tahmini'])
print(classification_report(y_test , y_pred_decisionEntropy))
decisionWEntropy = classification_report(y_test , y_pred_decisionEntropy,output_dict="true")
GinivsEntropy = { 'Decision Criterion': ['Gini','Entopy'],
         'Precision': [decisionWEntropy.get('weighted avg').get('precision') , decision.get('weighted avg').get('precision')],
         'Recall': [decisionWEntropy.get('weighted avg').get('recall') , decision.get('weighted avg').get('recall')],
         'F1-score': [decisionWEntropy.get('weighted avg').get('f1-score') , decision.get('weighted avg').get('f1-score')],
         'Accuracy': [decisionWEntropy.get('accuracy') , decision.get('accuracy')]
        }
df_GinivsEntropy = pd.DataFrame(GinivsEntropy, columns = ['Decision Criterion','Precision','Recall','F1-score', 'Accuracy'])
df_GinivsEntropy
KNN_komsular = { 'Neighbors': [],
         'Precision': [],
         'Recall': [],
         'F1-score': [],
         'Accuracy': []
        }
df_KNN_komsular = pd.DataFrame(KNN_komsular, columns = ['Neighbors','Precision','Recall','F1-score', 'Accuracy'])
for x in range(2,16):
    neigh = KNeighborsClassifier(n_neighbors=x)
    neigh.fit(X_train, Y_train)
    y_pred =  neigh.predict(X_test)
    knn_report = classification_report(Y_test , y_pred,output_dict="true")
    df_KNN_komsular.loc[x] =[x,knn_report.get('weighted avg').get('precision'),knn_report.get('weighted avg').get('recall'),knn_report.get('weighted avg').get('f1-score'),knn_report.get('accuracy')]
    
df_KNN_komsular
sns.barplot(x="Neighbors",y="Accuracy",data=df_KNN_komsular)
sns.barplot(x="Neighbors",y="F1-score",data=df_KNN_komsular)
sns.barplot(x="Neighbors",y="Recall",data=df_KNN_komsular)
sns.barplot(x="Neighbors",y="Precision",data=df_KNN_komsular)