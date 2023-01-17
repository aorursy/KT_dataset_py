import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns



from subprocess import check_output
startups = pd.read_csv("../input/50-startups/50_Startups.csv")

df = startups.copy()
df.head()
df.info()
df.shape #Datasetimiz 50 gözlem ve 5 öznitelikten oluşuyor
df.isna().sum() #Eksik verimiz bulunmuyor
df.corr() 

#Korelasyon katsayısı +1.00'a yaklastıkca pozitif iliski olur.

#En güçlü pozitif ilişki "R&D Spend" ile "Profit" arasındadır.
sns.regplot(df["R&D Spend"], df["Profit"], ci = None); #lineer pozitif bir ilişki olduğunu görüyoruz
corr = df.corr() 

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
#Isı haritası grafiğinde Profit ile #R&D Spend arasındaki pozitif yönlü ilişkinin şiddetinin yüksek olduğu görülmektedir
sns.scatterplot(x = "R&D Spend", y = "Profit", data = df);

#Aralarındaki pozitif ilişkiyi daha iyi görebiliyoruz

#Aralarında ilişkiye bakınca R&D Spend ve Profit doğru orantılı olarak artmaktadır 
df.hist(figsize = (15,15))
df.describe().T

#Marketing Spend en yüksek standart sapmaya sahiptir.

#Administration en küçük standart sapmaya sahiptir.

#Birçok veri ortalamaya yakın ise standart sapma düşük olacaktır.

#Birçok veri ortalamaya uzak ise standart sapma yüksek olacaktır.



#Administration en düzenli, Marketing Spend en düzensiz dağıldığı görülür.



#Bütün özniteliklerin standart sapması ortalamaya göre düşüktür. Bu da verilerin ayırt ediciliği düşüktür ve grup homojendir
df["State"].unique()
df['State'] = pd.Categorical(df['State'])
dfDummies = pd.get_dummies(df['State'], prefix = 'State')

#State'a dair kategori isimlerinini önüne State ekledim
dfDummies
df.drop(["State"], axis = 1, inplace = True)

dfDummies.drop(["State_New York"], axis = 1, inplace = True)
df = pd.concat([df, dfDummies], axis=1)

df
X = df.drop("Profit", axis = 1)

y = df["Profit"]
X
y
from sklearn.model_selection import train_test_split

#train test modellerini oluştumak için gerekli olacak.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)

#y değişkeni 0dan 49a büyükten küçüğe sıralı gibi eğer random_state kullanmazsak büyük değerleri öğretir küçük değerleri test ederiz ama biz bunu karışık yapmasını istiyoruz.

X_train
X_test
y_train
y_test
from sklearn.linear_model import LinearRegression



lr= LinearRegression()
model = lr.fit(X_train, y_train)
y_pred = model.predict(X_test)
df_sonuc=pd.DataFrame({'Gercek': y_test,'Tahmin Edilen': y_pred})

df_sonuc
from sklearn.metrics import mean_absolute_error



MAE = mean_absolute_error(df_sonuc["Gercek"], df_sonuc["Tahmin Edilen"])

MAE

#Artıkların mutlak değerleri alınır. 

#Ortalama hatayı temsil eder
from sklearn.metrics import mean_squared_error



MSE = mean_squared_error(df_sonuc["Gercek"], df_sonuc["Tahmin Edilen"])

MSE

#Artıkların kare toplamının örnek sayısına bölünmesi ile elde edilir. Burada kalıntılar, gözlem değerleri ile tahmin değerlerinin farkından oluşur. 

#0 değerine yaklaştıkça modelin tahmin yeteneği daha iyidir.

#Gözlem başına hata payını ölçmeye yarar
import math

RMSE = math.sqrt(MSE)

RMSE

#MSE'nin kare köküne göre hata oranıdır

#0 değerine yaklaştıkça modelin tahmin yeteneği daha iyidir.
model.score(X,y)

# bağımsız ve bağımlı öznitelikleri parametre olarak alır ve RSquared değerini döner.

# R squared değeri 1'e ne kadar yakınsa modeldeki bağımsız değişkenlerin bağımlı değişkeni ifade edebilme oranı o kadar iyidir.
import statsmodels.api as sm



stmodel = sm.OLS(y, X).fit()

stmodel.summary()
X = df[['R&D Spend', 'Administration', 'Marketing Spend']]

y = df["Profit"]
X
y
from sklearn.model_selection import train_test_split

#train test modellerini oluştumak için gerekli olacak.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)

#y değişkeni 0dan 49a büyükten küçüğe sıralı gibi eğer random_state kullanmazsak büyük değerleri öğretir küçük değerleri test ederiz ama biz bunu karışık yapmasını istiyoruz.

model_tekrar = lr.fit(X_train, y_train)
y_pred = model_tekrar.predict(X_test)
df_sonucTekrar=pd.DataFrame({'Gercek': y_test,'Tahmin Edilen': y_pred})

df_sonucTekrar
from sklearn.metrics import mean_absolute_error



MAE = mean_absolute_error(df_sonucTekrar["Gercek"], df_sonucTekrar["Tahmin Edilen"])

MAE

#Artıkların mutlak değerleri alınır. 

#Ortalama hatayı temsil eder
from sklearn.metrics import mean_squared_error



MSE = mean_squared_error(df_sonucTekrar["Gercek"], df_sonucTekrar["Tahmin Edilen"])

MSE

#Artıkların kare toplamının örnek sayısına bölünmesi ile elde edilir. Burada kalıntılar, gözlem değerleri ile tahmin değerlerinin farkından oluşur. 

#0 değerine yaklaştıkça modelin tahmin yeteneği daha iyidir.

#Gözlem başına hata payını ölçmeye yarar
import math

RMSE = math.sqrt(MSE)

RMSE

#MSE'nin kare köküne göre hata oranıdır

#0 değerine yaklaştıkça modelin tahmin yeteneği daha iyidir.
model_tekrar.score(X,y)