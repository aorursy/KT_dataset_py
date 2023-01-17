# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report ,confusion_matrix
import seaborn as sns
import missingno 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
df = pd.read_csv("../input/tabletpc-priceclassification/tablet.csv")
df.dataframeName = "tablet.csv"
df.head()

df["MikroislemciHizi"].value_counts()
#0.5 bir yüklenme var 3.0 da düşüklük var geriye kalan veriler dengeli
df["BataryaGucu"].value_counts()
df["BataryaOmru"].value_counts()
df["CozunurlukYükseklik"].value_counts()
df["FiyatAraligi"].value_counts()
df.info()
df.corr()
#8  object türünde verimiz var onları ilerleyen zamanlarda sayısallaştırma işlemleri yapıp dahil edecez

corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values
          );

sns.scatterplot(x = "CozunurlukGenislik", y = "CozunurlukYükseklik",sizes=(40, 400),hue='FiyatAraligi', data = df);
sns.scatterplot(x = "ArkaKameraMP", y = "OnKameraMP",sizes=(40, 400),hue='FiyatAraligi', data = df);
sns.jointplot(x = "BataryaOmru", y = "BataryaGucu", data = df, color="blue"); 
sns.swarmplot(x = 'BataryaOmru', y = 'BataryaGucu', data = df)
sns.lmplot(x='MikroislemciHizi', y='RAM', fit_reg = False, hue = 'FiyatAraligi',data=df)
sns.boxplot(data = df)
stats_df = df.drop(['BataryaGucu', 'MikroislemciHizi', 'OnKameraMP','ArkaKameraMP','CozunurlukYükseklik','BataryaOmru','RAM'], axis = 1)
sns.boxplot(data = stats_df, notch = True, linewidth = 2.5, width = 0.50)
sns.set_style('whitegrid')
sns.violinplot(x = 'Kalinlik', y = 'Agirlik', data = df)
sns.distplot(df.CekirdekSayisi)
sns.countplot(x='DahiliBellek', data=df)
df.describe().T
missingno.matrix(df,figsize=(20, 10));
df.isna( ).sum( ) 
df['OnKameraMP'].mean()

df['OnKameraMP']= df['OnKameraMP'].fillna(df['OnKameraMP'].mean())


df['RAM']=df['RAM'].fillna(df['RAM'].mean())
# eksik veri sayısı az olduğundan dolayı ortalama değerler ile doldurmak istedim
df.isnull().sum().sum()
df["FiyatAraligi"].unique()
pd.get_dummies(df["Renk"])
df_Bluetooth=pd.get_dummies(df["Bluetooth"])
df_Bluetooth.columns = ['BluetoothVar','BluetoothYok']
df=pd.concat([df,df_Bluetooth],axis=1)
df_4G=pd.get_dummies(df["4G"])
df_4G.columns = ['4GVar','4GYok']
df=pd.concat([df,df_4G],axis=1)
df_3G=pd.get_dummies(df["3G"])
df_3G.columns = ['3GVar','3GYok']
df=pd.concat([df,df_3G],axis=1)
df_CiftHat=pd.get_dummies(df["CiftHat"])
df_CiftHat.columns = ['CiftHatVar','CiftHatYok']
df=pd.concat([df,df_CiftHat],axis=1)
df_Dokunmatik=pd.get_dummies(df["Dokunmatik"])
df_Dokunmatik.columns = ['DokunmatikVar','DokunmatikYok']
df=pd.concat([df,df_Dokunmatik],axis=1)
df_WiFi=pd.get_dummies(df["WiFi"])
df_WiFi.columns = ['WiFiVar','WiFiYok']
df=pd.concat([df,df_WiFi],axis=1)
df.drop(['WiFi','Bluetooth','3G','4G','CiftHat','Dokunmatik'], axis=1, inplace=True)
label_encoder = preprocessing.LabelEncoder()
df['Renk'] = label_encoder.fit_transform(df['Renk'])
df.drop(["WiFiYok","BluetoothYok","3GYok","4GYok","CiftHatYok","DokunmatikYok"], axis=1, inplace=True)
df.corr()
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values
          );

sns.scatterplot(x = "4GVar", y = "3GVar", hue = "FiyatAraligi", data = df);
sns.barplot(x="FiyatAraligi",y="RAM",data=df)
X = df.drop("FiyatAraligi",axis=1)
y = df["FiyatAraligi"]
X
y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)
df.info()
y_test
X_train
X_test
y_train
gnb = GaussianNB()
modelgnb=gnb.fit(X,y)
y_pred = modelgnb.predict(X_test)
y_pred
accuracy_score(y_test,y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)
karmasiklik_matrisi
F1Score = f1_score(y_test, y_pred, average = 'weighted')  
F1Score
RecallScore = recall_score(y_test, y_pred, average='weighted')
RecallScore
PrecisionScore = precision_score(y_test, y_pred, average='weighted')
PrecisionScore
print(classification_report(y_test, y_pred)) 
cross_val_score(modelgnb, X, y, cv = 7)
tree = DecisionTreeClassifier()
modeltree = tree.fit(X_train,y_train)
modeltree
y_pred = modeltree.predict(X_test)
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)
print(karmasiklik_matrisi)
F1Score = f1_score(y_test, y_pred, average = 'weighted')  

RecallScore = recall_score(y_test, y_pred, average='weighted')
PrecisionScore = precision_score(y_test, y_pred, average='weighted')
print(classification_report(y_test, y_pred)) 
accuracy_score(y_test, y_pred)
tree = DecisionTreeClassifier(criterion='entropy')
modeltree = tree.fit(X_train, y_train)
y_pred1 = modeltree.predict(X_test)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred1)
print(karmasiklik_matrisi)
PrecisionScore = precision_score(y_test, y_pred1, average='weighted')
RecallScore = recall_score(y_test, y_pred1, average='weighted')
F1Score = f1_score(y_test, y_pred1, average = 'weighted')  
print(classification_report(y_test, y_pred1))
accuracy_score(y_test, y_pred1)
knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)
y_pred2 = knn_model.predict(X_test)
accuracy_score(y_test, y_pred2)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred1)
print(karmasiklik_matrisi)
F1Score = f1_score(y_test, y_pred2, average = 'weighted')  
F1Score
RecallScore = recall_score(y_test, y_pred2, average='weighted')
RecallScore
PrecisionScore = precision_score(y_test, y_pred2, average='weighted')
PrecisionScore
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred2))
KomsuSayisi = [] 

for k in range(2,15,1): 
    knnKomsuSayisi = KNeighborsClassifier(n_neighbors = k)
    knnKomsuSayisi.fit(X_train,y_train)
    KomsuSayisi.append(knnKomsuSayisi.score(X_test, y_test))

plt.plot(range(2,15,1),KomsuSayisi)
plt.xlabel("En Yakın Komşu Sayısı =K")
plt.ylabel("Doğruluk Oranı")
plt.show()
knn_Komsu = KNeighborsClassifier(n_neighbors =13)
knn_Komsu = knn_Komsu.fit(X_train, y_train)
cross_val_score(knn_model, X_test, y_test, cv = 5)
PrecisionScore = precision_score(y_test, y_pred, average='weighted')
PrecisionScore
PrecisionScore = precision_score(y_test, y_pred, average='weighted')
PrecisionScore
PrecisionScore = precision_score(y_test, y_pred, average='weighted')
PrecisionScore
print(classification_report(y_test, y_pred))