import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#Udemy "Python for Data Science" kursu için geliştirdiğim model
import pandas as pd
df = pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv") #veriyi okutalım
df.isnull().sum() #kayıp veri denetimi
df.describe().transpose() #veriye genel bakış
plt.figure(figsize=(10,6))

sns.distplot(df["price"]) #fiyat dağılımı
sns.countplot(df["bedrooms"]) #yatak odasına göre dağılım
df.corr()["price"].sort_values() #fiyat ile korelasyonu olan değişkenler
plt.figure(figsize=(10,5))

sns.scatterplot(x="price",y="sqft_living",data=df) #korelasyonu en fazla olan sqft_living değişkeninin price ile olan ilişkisi
plt.figure(figsize=(10,6))

sns.boxplot(x="bedrooms",y="price",data=df) #yatak odası sayısı ile fiyat arasındaki ilişki
plt.figure(figsize=(12,8))

sns.scatterplot(x="price",y="long",data=df) #fiyatın boylamlara göre dağılımı
plt.figure(figsize=(12,8))

sns.scatterplot(x="price",y="lat",data=df) #fiyatın enlemlere göre dağılımı
plt.figure(figsize=(12,8))

sns.scatterplot(x="long",y="lat",data=df,hue="price") #en pahalı evleri görüntülemek için oluşturduğumuz plot
#veride çok pahalı birkaç ev olduğu var ve bunlar normalliği bozuyor

#plot'un daha iyi anlaşılması için yüzde 1'lik bölümü çıkaracağız. bunun için ilk önce en pahalı evleri görelim

df.sort_values("price",ascending=False).head(20)
#veride bulunan toplam ev sayısı

len(df)
#bu evlerin yüzde birlik bölümü

len(df)*0.01
#normalliği bozan evleri daha rahat görebiliriz

df.sort_values("price",ascending=False)
#veride en pahalı evlerin yüzde birini çıkaralım

non_top_1_perc = df.sort_values("price",ascending=False).iloc[217:]
#şimdi veriye tekrar göz atalım

#plotın daha anlaşılır gözükmesi için alpha ve palette değerleri girelim

plt.figure(figsize=(12,8))

sns.scatterplot(x="long",y="lat",data=non_top_1_perc,hue="price",edgecolor=None,alpha=0.2,palette="RdYlGn")
#exploratory data analysisi devam ettirelim

#denize nazır evlerin fiyat farkına bakalım

sns.boxplot(x="waterfront",y="price",data=df)
#modeli yanlış beslememek için anlamsız sütunları çıkaralım

df = df.drop("id",axis=1)
df = df.drop("zipcode",axis=1)
#tarih sütununu düzeltelim

df["date"] = pd.to_datetime(df["date"])
#tarih sütununu ay ve yıl diye iki sütuna ayıralım

df["year"] = df["date"].apply(lambda date: date.year)

df["month"] = df["date"].apply(lambda date: date.month)
#veriye tekrar bakalım

df.head()
#fiyatın yıllar içinde sürekli olarak yükseldiğini görebiliriz

df.groupby("year").mean()["price"].plot()
#tarih sütununu ikiye ayırdığımız için artık buna ihtiyacımız yok

df = df.drop("date",axis=1)
X = df.drop("price",axis=1).values

y = df["price"].values
#Scikit Learn'den veri ayırma işlemini çağıralım

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
#SciKit'ten ölçekleyiciyi çağıralım

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
#TF ve Keras'tan model ve katmanları çağıralım

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
#train verisinin şeklini inceleyelim

X_train.shape
#modeli ve katmanları oluşturalım

model = Sequential()



model.add(Dense(19,activation="relu"))

model.add(Dense(19,activation="relu"))

model.add(Dense(19,activation="relu"))

model.add(Dense(19,activation="relu"))



model.add(Dense(1))



model.compile(optimizer="adam",loss="mse")
#400 epoch ve 128 batch'lik modeli eğitelim

model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=128,epochs=400)
#kayıp değere bakalım

losses = pd.DataFrame(model.history.history)

losses.plot()
#under veya overfit gözükmüyor

#şimdi tahminlere bakalım

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
predictions = model.predict(X_test)
mean_squared_error(y_test,predictions)
mean_absolute_error(y_test,predictions)
explained_variance_score(y_test,predictions)
#veri ve tahminler arasındaki ilişkiye bakalım

plt.figure(figsize=(12,6))

plt.scatter(y_test,predictions)

plt.plot(y_test,y_test,"r")
#veriden bir satır alıp modelin fiyatı tahmin etmesini isteyelim

yeni_ev = df.drop("price",axis=1).iloc[0]
yeni_ev = scaler.transform(yeni_ev.values.reshape(-1,19))
model.predict(yeni_ev)
#önceki hücrede model evin fiyatını tahmin etti. şimdi verideki gerçek fiyat ile karşılaştıralım

df.head(1)
#modelin tahmin ettiği fiyat 281747

#evin gerçek fiyatı 221900