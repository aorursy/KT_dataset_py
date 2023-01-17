import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt
startups = pd.read_csv("../input/50-startups/50_Startups.csv") # Pandas aracılığı ile datasetimizi tanıttık

df = startups.copy() # kopyasını aldık bu yüzden herhangi bir sorun oluşursa 
df.head(5)
df.info()
df.shape
df.isna().sum() # eksik verimiz yok
df.corr()
corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
sns.scatterplot(x = "R&D Spend", y = "Profit", data = df, color="blue");
df.hist(bins=12)

# sns.pairplot(df) gibi plotlar ile de farklı çözüm yöntemleri vardır

# Histogram çizdirme işlemi Matplotlib kütüphanesinde yer almaktadır.
df.describe().T
df.State.unique()
df_State = pd.get_dummies(df["State"])
df_State.head()
df_State.columns = ['California','Florida','New York']

df_State.head() # İsimleri değiştirebiliyoruz fakat aynı isimleri verdim ki ilerideki işlemlerde sorun oluşmasın.
df = pd.concat([df, df_State], axis = 1)

df.drop(["California", "State"], axis = 1, inplace = True) # California State'ini dropluyoruz yani çıkarıyoruz.

# Bunu yaparken df üzerinden işlem yapıyoruz çünkü df gerçek veri setimizin kopyasıdır.

# İkinci çalıştırmada hata vericektir çünkü California'yı bulamayacaktır.
df.head()
Y = df.iloc[:,3].values

X = df[['R&D Spend', 'Administration', 'Marketing Spend']]

# X = df.iloc[:,:-3].values de yazılabilirdi.3 sütunu hariç tut demek

# X = df.drop("Profit", axis = 1) de yazılabilirdi.

# X bağımlı değişkeni açıklamakta kullanılan değişken veya değişkenler

# Y açıklamak istenilen hedef değişken.

# x bağımsız değişken ve profit hariç değişkenler, y ise profit olmalıdır

# Basit regresyon bir tane bağımlı değişken bir tane de bağımsız değişkenden oluşmaktadır

# Çoklu regresyon ise, bir adet bağımlı değişken ve birden fazla bağımsız değişkenin bir arada bulunduğu modeldir
X
Y
from sklearn.model_selection import train_test_split # Değişkenlerimizi eğitim ve test olmak üzere ikiye bölebilmek için gerekli

# kütüphane

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 42, shuffle=1)

# x ve y verilerinin 2/3 ü eğitim için  1/3 ü test için yollanacağını ve bu verilerin karışık gideceğini yazdık
X_train
X_test
Y_train
Y_test
from sklearn.linear_model import LinearRegression # Doğrusal(Lineer) Regresyon için gerekli kütüphane

model=LinearRegression()
model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)
df_gozlem = pd.DataFrame({"Gerçek" : Y_test, "Tahmin Edilen" : Y_pred})



df_gozlem
import sklearn.metrics as metrics # MAE, MSE, RMSE her birine erişebileceğimiz kütüphane



mae = metrics.mean_absolute_error(Y_test, Y_pred)

mse = metrics.mean_squared_error(Y_test, Y_pred)

rmse = np.sqrt(mse) # Rmse = Kök Mse



print("Mean Absolute Rrror(MAE):",mae) #  Ortalama Mutlak Hata

print("Mean Squared Error (MSE):", mse) # Ortalama Kare Hata

print("Root Mean Squared Error (RMSE):", rmse) # Kök Ortalama Kare Hata (MSE'nin kök alınmış hali)
model.score(X_train, Y_train)
import statsmodels.api as sm # gerekli kütüphanemiz
model = sm.OLS(Y, X).fit()
print(model.summary())