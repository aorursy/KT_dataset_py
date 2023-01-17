import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
df = pd.read_csv("../input/50_Startupss.csv").copy()

df.head()
df.info()
df.shape   # 5 öznitelik 50 adet gözlem mevcut
df.isnull().sum()#görünürde hiç boş gözlem yok

corr = df.corr()#kar ile arge harcamaları arasında pozitif olarak çok güçlü bir ilişki var

corr #pazarlama harcamaları ile arge harcamaları arasında pozitif yönde kısmen güçlü bir ilişki var
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)

#renklendirmeler beyaza yaklaştıkça aralarındaki pozitif korelasyon katsayısı artmış

#yönetim ile pazarlama harcamaları arasında neredeyse hiç bir ilişki yok diyebiliriz
sns.scatterplot(x="R&D Spend",y="Profit",data = df)#aralarında doğrusal bir korelasyon söz konusu
sns.pairplot(df)#pairplot sayısal değişkenleri hem histogram hem parçacık olarak çizmemizi sağladı.
df.describe().T#market ve araştırma geliştirme harcaması olmayan şirketler görünüyor

#standart sapmaya göre harcamalar oldukça değişkenlik gösteriyor
df["State"].unique()
df["State"]= pd.Categorical(df["State"])

dfDummies = pd.get_dummies(df["State"])

dfDummies.columns = ['California','Florida','New York']

#verileri makine öğrenmesine sokacağımız için sayısal verilere çevirmeliyiz.Makine için sadece sayılar anlam taşır.

#kategorik şekilde yazdık birbirlerine üstünlüğünün olmadığını makineye anlatmamızın bir yolu olmuş oldu.



dfDummies.head(5)
df = pd.concat([df,dfDummies],axis=1)#iki listeyi birbirine bağladık

df.head()
df= df.drop("State", axis=1)#artık bu kolona ihtiyacımız olmadığından kolonu kaldırdık.

df=df.drop("New York",axis=1)#1 ve 0 olarak tanımlanan 3 farklı kategorik değerimiz olduğu ve dolayısıyla 2 kolonunun değerini bilirsek diğerinin sonucu bilinebileceği için bu kolonu çıkartabiliriz.
df.head()
y = df['Profit']#bağımlı değişkenimiz kar

X = df.drop(['Profit'], axis=1)#bağımsız değişkenlerimiz

#y = df.iloc[:,3].values

#X = df[['R&D Spend', 'Administration', 'Marketing Spend']]
y
X
from sklearn.model_selection import train_test_split #verileri eğitime ve teste tabi tutmak amacı ile eklediğimiz kütüphane

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)#verilerimizin 1/4 ü test edilecek şekilde oranlandı. Kalan kısmı ile makine öğrenmesi yapılacak.
X_train
X_test
y_train
y_test
from sklearn.linear_model import LinearRegression 

model=LinearRegression()#artık asıl amacımız olan doğrusal regresyon için kütüphanemizi ekleyip modelimizi oluşturuyoruz. 
model.fit(X_train,y_train)#ayırdığımız verileri modeli eğitmek için gönderiyoruz.
y_pred = model.predict(X_test)#modelimizdeki bağımsız değişkenleri predict fonksiyonuna vererek bağımlı değişkenimizi tahmin ediyoruz.
df_gozlem = pd.DataFrame({"Gerçek deger" : y_test, "Tahmini deger" : y_pred})



df_gozlem
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, precision_score
from sklearn import metrics

print("mean squared error (MSE)" , metrics.mean_squared_error(y_test,y_pred))#buradaki değerler 1 e ne kadar yakınsa model bizim için o kadar iyi demektir

print("Root mean squared error (RMSE)" , np.sqrt(metrics.mean_squared_error(y_test,y_pred)))#ne kadar düşükse o kadar başarılıdır.
import statsmodels.api as sm
model = sm.OLS(y, X).fit()
print(model.summary())