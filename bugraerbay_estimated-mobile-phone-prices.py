import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv('../input/tabletpc-priceclassification/tablet.csv')

df.head(15)
df.shape
df.describe().T

df.corr()
df.hist()
sns.scatterplot(df["RAM"], df["FiyatAraligi"]);
df.groupby("FiyatAraligi")[["RAM"]].aggregate("mean")

#RAM ORTALAMASI ILE FIYATARALIGI ILISKISI - Doğrudan bir ilişki var.
df.groupby("FiyatAraligi")[["MikroislemciHizi"]].aggregate("mean")

#Mikro işlemci hızı ile fiyat aralığının ortalaması arasındaki ilişki.
def countplot(baslik): 

    sns.countplot(x=baslik, data=df)

    plt.xticks(rotation=100)
countplot("FiyatAraligi")

#Fiyat aralığı - adet grafiği
countplot("Renk")

#Renk - adet grafiği
df.isnull().sum()

#Boş değerlerimizin sayısını saptıyoruz.
df[df.isnull().any(axis=1)]

#Eksik olan değerlerimize erişiyoruz.
df.groupby("FiyatAraligi")["RAM"].mean()

#Boş değer bulunan index'in,bulunduğu fiyat aralığına göre ortalamasını buluyoruz.



df["RAM"].fillna(df.groupby("FiyatAraligi")["RAM"].transform("mean"), inplace = True)

#Boş değeri, bulunduğu fiyat aralığındaki ortalamaya göre doldurduk.
df.groupby("FiyatAraligi")["OnKameraMP"].mean()

#Boş değer bulunan index'in,bulunduğu fiyat aralığına göre ortalamasını buluyoruz.



df["OnKameraMP"].fillna(df.groupby("FiyatAraligi")["OnKameraMP"].transform("mean"),inplace = True)

#Boş değeri, bulunduğu fiyat aralığındaki ortalamaya göre doldurduk.
df.isnull().sum().sum()

#Boş değerlerimizi kalıcı olarak doldurduk.
df['Renk']
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

df['Renk'] = label_encoder.fit_transform(df['Renk'])

df['Renk'].unique()

df['FiyatAraligi'] = label_encoder.fit_transform(df['FiyatAraligi'])

df['FiyatAraligi'].unique()

#Kategorik verilerimize sayısal değerler atadık.
df['CiftHat'] = label_encoder.fit_transform(df['CiftHat']) #Var:0 Yok:1

df['4G'] = label_encoder.fit_transform(df['4G']) #Var:0 Yok:1

df['3G'] = label_encoder.fit_transform(df['3G']) #Var:1 Yok:0

df['Bluetooth'] = label_encoder.fit_transform(df['Bluetooth']) #Var:0 Yok:1

df['WiFi'] = label_encoder.fit_transform(df['WiFi']) #Var:0 Yok:1

df['Dokunmatik'] = label_encoder.fit_transform(df['Dokunmatik'])

#2 değer alabilecek sayılarımızı binarize işlemi gerçekleştirdik. 

# 0 - Normal , 1 - Pahalı, 2-Ucuz, 3-Çok ucuz
df.head()
x = df.drop("FiyatAraligi", axis=1)

y = df["FiyatAraligi"]
x.head()
y.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/4, random_state = 42)
x_train.head()
x_test.head()
y_train.head()
y_test.head()
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train,y_train)

nb.score(x_train,y_train)
lm.fit(x_train,y_train)

lm.score(x_test,y_test)
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(random_state=42)

dtree.fit(x_train,y_train)
dtree.score(x_test,y_test)