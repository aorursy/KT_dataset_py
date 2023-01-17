import numpy as np #numpy kütüphanemi dahil ettim.

import pandas as pd #pandas kütüphanemi dahil ettim.

from matplotlib import pyplot as plt #matplotlib kütüphanemi dahil ettim.

import seaborn as sns #seaborn kütüphanemi dahil ettim.
startups = pd.read_csv("../input/50-startupscsv/50_Startups.csv",sep = ",") #Verilen csv dosyasında veriler virgülle ayrıldıkları için sep'i virgül olarak seçtim.

df=startups.copy() #İşlem yaparken asıl dosyamın etkilenmemesi için kopyasını df değişkenime atadım.
df.head()
df.info() #Veri çerçevem 5 sütundan oluşmaktadır.Bunlardan State hariç hepsinin datatype'ı float64 iken State'in datatype'ı Objecttir.
df.shape #Veri çerçevesi 50 gözlem 5 öznitelikten oluşmaktadır.
df.isnull().sum() #Hiçbir özniteliğimde eksik veri bulunmamaktadır.
df.corr() #Korelasyon matrisine göre en güçlü ilişki Profit ve R&D Spend arasındadır.Çünkü korelasyonları 0.972900'dır.Korelasyon 1'e yaklatıkça ilişki mükemmelleşir.En zayıf ilişki ise Marketing Spend ve Administration arasındadır.
corr=df.corr() #Isı haritasında da Profit ve R&D Spend arasında mükemmel bir ilişki olduğunu görebiliyoruz.

sns.heatmap(corr,

           xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
sns.scatterplot(x=df["R&D Spend"],y=df["Profit"]); #R&D Spend artarken Profitte artıyor aralarındaki ilişkinin yüksek olmasından dolayı.
df.hist(figsize =(13,8), color = "purple")

plt.show()
df.describe().T
df["State"].unique() #State'e ait benzersiz olan değerler New York,California,Florida'dır.Bunlarda object tipindedir.
pd.get_dummies(df["State"])
df['State'] = pd.Categorical(df['State'])
dfDummies = pd.get_dummies(df['State'], prefix = 'State')
dfDummies
df.drop(["State"], axis = 1 , inplace = True) #State özniteliğini siliyorum.

df = pd.concat([df, dfDummies], axis=1) #Dummy olarak yarattığımız State'leri ekliyoruz.

df #Görüldüğü üzere Dummy olarak yarattığımız başlarına State koyduğumuz özniteliklerimizi tabloya concat ettik
df.drop(["State_New York"],axis=1,inplace=True) #State_New York sütununu kaldırıyorum.

df #Görüldüğü gibi State_New York sütununu kaldırdım.
x = df.drop("Profit", axis = 1) #x bağımsız değişkeni ifade ederken y bağımlı değişkeni ifade etmektedir.Bize verilen bilgilerde Profit bağımlı değişken olduğu için y'ye atarken geri kalan özniteliklerimi x'e atadım.

y = df["Profit"]
x #Bağımsız değişkenlerimi yazdırıyorum.
y #Bağımlı değişkenimi yazdırıyorum.
from sklearn.model_selection import train_test_split #Kütüphanemi dahil ediyorum.

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=42) #x,y train ve x,y test'lerden 4 parça oluşturuyorum.
x_train #x train değişkenimi kontrol ediyorum.
x_test #x test değişkenimi kontrol ediyorum.
y_train #y train değişkenimi kontrol ediyorum.
y_test #y test değişkenimi kontrol ediyorum.
from sklearn.linear_model import LinearRegression #LinearRegression' çekirdeğime dahil ediyorum.

model = LinearRegression()
model.fit(x_train,y_train) #Eğitim verilerimi modele veriyorum.Bunlardan x_train bağımsız y_train bağımlı olan değişkenimdir.
y_pred=model.predict(x_test)
df_veri = pd.DataFrame({'Gerçek Değer': y_test, 'Tahmini Değer': y_pred})

df_veri
from sklearn.metrics import mean_absolute_error #MAE için kütüphanemi ekliyorum.

MAE = mean_absolute_error(y_test, y_pred) #Ortalama mutlak hatam 6566.642122870217'dır.Bu değerin daha düşük olması iyi olanıdır.

MAE #Ortalama mutlak hatamı görüntülüyorum.
from sklearn.metrics import mean_squared_error #MSE için kütüphanemi ekliyorum.

MSE = mean_squared_error(y_test, y_pred)#MSE'yi makine öğrenmesi modelinin performansını ölçmekte kullanırız.

MSE #MSE değerimi görüntülüyorum.
import math #RMSE için kütüphanemi ekliyorum.

RMSE = math.sqrt(MSE) #RMSE Tahmin hatalarımın standart sapmasını vermektedir.Benim tahmin hatalarımın standart sapması 8640.539333502398'dır. 

RMSE #RMSE değerimi görüntülüyorum.
model.score(x_train,y_train)
import statsmodels.api as stat

stmodel = stat.OLS(y_train, x_train).fit()

stmodel.summary() #Burada tablo verimiz hakkında birçok bilgi içermektedir.Bağımlı değişkenimiz Profit,R-squared değeri 0.988 gibi bilgileri elde ediyoruz.Güvenilirliğim 0.05'tir.

#Anlamlı olup olmadıklarını P>|t| değerlerine bakarak anlıyoruz.Eğer bu değer 0.05'den düşükse bu veriler anlamlıdır.Bizim modelimizde R&D Spend ve Administration değerleri 0.000 olduğu için bizim için çok anlamlı veirlerdir.Bunun yanı sıra Marketing Spend'te önceki iki veri gibi çok anlamlı olmasada yinede anlamlıdır.Çünkü değeri 0.05'ten küçüktür.