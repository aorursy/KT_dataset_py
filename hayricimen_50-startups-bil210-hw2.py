import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

startups = pd.read_csv("../input/50-startups/50_Startups.csv") # Pandas kütüphanesi ile datasetimizi startups'a aktardık

df = startups.copy()
df.head(5)
df.info()
df.shape 

df.isna().sum() 
df.corr()



sns.heatmap(df.corr());
sns.scatterplot(x = "R&D Spend", y = "Profit", data = df ); 

df.hist(figsize =(13,10))

plt.show()
df.describe().T

df["State"].unique()
df_State = pd.get_dummies(df["State"])
df_State.head(10)
df_State.columns = ['California','Florida','New York']
df_State.head(10)
df.drop(["State"], axis=1 , inplace =True)

df=pd.concat([df,df_State],axis=1)
df.head()
df.drop(["New York"], axis=1, inplace = True)

df.head()
X = df.drop(["Profit"] , axis = 1) # Bağımsız değişkenler listesi

Y = df["Profit"] # Bağımlı değişkenler listesi
X  # Bağımsız değişkenler listesini yazdıralım

Y  # Bağımlı değişkenler listesini yazdıralım
from sklearn.model_selection import train_test_split#train test modellerini oluştumak için gerekli olacaktır.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, train_size = 0.4, random_state = 0, shuffle=1)

# Verilerin test ve train için kullanılak miktarı seçtik (4/10),

# Verilerin karışık gitmesi için "shuffle=1" yazdık,
X_train
X_test
Y_train
Y_test
from sklearn.linear_model import LinearRegression #Lineer(Doğrusal) regresyonu uygulamak için gerekli kütüphaneyi ekliyoruz.

model=LinearRegression()  
model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)

Y_pred # Y_pred'in içeriğini görelim
df = pd.DataFrame({"Gerçek Veriler" : Y_test, "Tahmin Edilen" : Y_pred,"Hata Payı" : abs(Y_test - Y_pred)})



df   

# Gerçek Veriler ile Tahminler arsındaki farklarında mutlak değerlerini yazdırıyoruz
import sklearn.metrics as metrics



mae = metrics.mean_absolute_error(Y_test, Y_pred)

mse = metrics.mean_squared_error(Y_test, Y_pred)

rmse = math.sqrt(mse)



print("Mean Absolute Error(MAE):",mae) #  Hata Mutlak Oralaması

print("Mean Squared Error (MSE):", mse) # Hata Kare Ortalaması 

print("Root Mean Squared Error (RMSE):", rmse) # Hata kareler ortalamasının karekökü  
print("R Squared=", model.score(X_train, Y_train))
import statsmodels.api as sm

sttmodel = sm.OLS(Y, X).fit()

sttmodel.summary()