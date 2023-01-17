import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import missingno as msno 

dataset=pd.read_csv('../input/tablet/tablet.csv')

dataset.head()
dataset.info()
sns.pairplot(dataset,hue='FiyatAraligi')
sns.pointplot(y="DahiliBellek", x="FiyatAraligi", data=dataset)
labels = ["3G destekli",'3G desteksiz']

values=dataset['3G'].value_counts().values
fig1, ax1 = plt.subplots()

ax1.pie(values, labels=labels, autopct='%1.1f%%',shadow=True,startangle=90)

plt.show()
labels = ["4G-destekli",'4G desteksiz']

values=dataset['4G'].value_counts().values
fig1, ax1 = plt.subplots()

ax1.pie(values, labels=labels, autopct='%1.1f%%',shadow=True,startangle=90)

plt.show()
sns.boxplot(x="FiyatAraligi", y="BataryaGucu", data=dataset)
plt.figure(figsize=(10,6))

dataset['OnKameraMP'].hist(alpha=0.5,color='blue',label='ÖnKameraCözünürlükYükseklik')

dataset['ArkaKameraMP'].hist(alpha=0.5,color='red',label='ArkaKameraCözünürlükYükseklik')

plt.legend()

plt.xlabel('CozunurlukYükseklik')
sns.pointplot(y="BataryaOmru", x="FiyatAraligi", data=dataset)
data = pd.DataFrame(dataset)
dataset.isnull().sum().sum()
data.isnull().sum()

dataset["OnKameraMP"].unique()
dataset["RAM"].unique()
msno.matrix(dataset)
dataset.mean()
data.dropna()
data["OnKameraMP"].fillna(0 ,inplace = True)

data["RAM"].fillna(0 ,inplace = True)
varYok_mapping = {"Yok": 0, "Var": 1}

data['Bluetooth'] = data['Bluetooth'].map(varYok_mapping)

data['CiftHat'] = data['CiftHat'].map(varYok_mapping)

data['4G'] = data['4G'].map(varYok_mapping)

data['3G'] = data['3G'].map(varYok_mapping)

data['Dokunmatik'] = data['Dokunmatik'].map(varYok_mapping)

data['WiFi'] = data['WiFi'].map(varYok_mapping)



fiyat_mapping = {"Çok Ucuz": 0, "Ucuz": 1, "Normal": 2, "Pahalı": 3}

data['FiyatAraligi'] = data['FiyatAraligi'].map(fiyat_mapping)


data = pd.concat([data, pd.get_dummies(data['Renk'], prefix='Renk')], axis=1)

data = data.drop('Renk',axis=1)

data
sns.jointplot(x='RAM',y='FiyatAraligi',data=data,color="purple", kind='kde');
sns.jointplot(x='Agirlik',y='FiyatAraligi',data=data, color="red", kind='kde');
x=data.drop('FiyatAraligi',axis=1)
y=data['FiyatAraligi']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=101)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train,y_train)
nb.score(X_train,y_train)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train,y_train)
lm.score(X_test,y_test)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train,y_train)

error_rate = []

for i in range(2,15):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(2,15),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=5)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
logmodel.score(X_test,y_test)
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='entropy', random_state=1)
dtree.fit(X_train,y_train)
dtree.score(X_test,y_test)
feature_names=['BataryaGucu', 'Bluetooth', 'MikroislemciHizi', 'CiftHat', 'OnKameraMp', '4G',

       'DahiliBellek', 'Kalinlik', 'Agirlik', 'CekirdekSayisi', 'ArkaKameraMP', 'CozunurlukYükseklik',

       'CozunurlukGenislik', 'RAM', 'BataryaOmru', '3G', 'BataryaOmru', 'Dokunmatik',

       'WiFi', 'FiyatAraligi']
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train, y_train)
rfc.score(X_test,y_test)
y_pred=lm.predict(X_test)
plt.scatter(y_test,y_pred)
plt.plot(y_test,y_pred)
from sklearn.metrics import classification_report,confusion_matrix
pred = knn.predict(X_test)
print(classification_report(y_test,pred))
matrix=confusion_matrix(y_test,pred)

print(matrix)
plt.figure(figsize = (10,7))

sns.heatmap(matrix,annot=True)
data_test=pd.DataFrame(data)
data_test.head()
Fiyat_tahmini=knn.predict(X_test)
Fiyat_tahmini
data_test
data_test['FiyatAraligi']= pd.Series(Fiyat_tahmini)
data_test