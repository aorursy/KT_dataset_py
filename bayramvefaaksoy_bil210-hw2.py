import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



startups = pd.read_csv("../input/startups/50_Startups.csv")

df=startups.copy()
df.head()
df.info()
df.shape #50 gözlem ve 5 değişkenden oluşan bir datasetimiz var
df.isna().sum()#Hiç eksik verimiz bulunmamaktadır.
df.corr()#Profit ile r&d spend arasında güçlü pozitif yönlü bir ilişki bulunmaktadır.
corr=df.corr()

sns.heatmap(corr,

           xticklabels=corr.columns.values,

           yticklabels=corr.columns.values)
sns.scatterplot(x="R&D Spend",y="Profit",data=df)#burada da profitin r&d spende göre artışını görebiliyoruz.

df.hist(figsize = (15,15))



plt.show()
df.describe()#değişkenlerin istatiksel olarak önemli değerlerinin gösterimi.  
df["State"].unique()
df_state=pd.get_dummies(df["State"] , prefix="State")
df_state.columns=['Bursa','Ankara','İzmir']

df_state.head() # kolon isimlerini değiştirebiliyoruz. 
df_state.columns=['California','Florida','New York']

df_state.head()#Yine aynı isimleri koyuyoruz.i
df.drop(["State"], axis=1 , inplace =True)

df=pd.concat([df,df_state],axis=1)
df.head()



df.drop(["California"], axis=1, inplace = True)

df.head()
X = df.drop(["Profit"], axis=1)#Test ve eğitim yaptıracağımız değişkenlerimize değerlerini atıyoruz.Bağımsız değişkenler

Y = df["Profit"]#bize verilen diğer değişkenleri kullanarak kâr değişkenini hesaplayacağız.Bağımlı değişkenler 
Y
X
from sklearn.model_selection import train_test_split#train test modellerini oluştumak için gerekli olacaktır.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 2/5, random_state = 2, shuffle=1)#Parametreler;X ve Y arrayleri,test_size; 5'te 2'si test için olacak,5'te 3'ü train için olacak.Shuffle verilerimizin karışık bir şekilde train ve test değişkenlerine gideceğini belirtir. 
X_train
X_test
Y_train
Y_test
from sklearn.linear_model import LinearRegression#Basit Doğrusal regresyonu uygulamak için gerekli.

model=LinearRegression()
model.fit(X_train, Y_train)
y_pred=model.predict(X_test)
df = pd.DataFrame({"Gerçek" : Y_test, "Tahmin Edilen" : y_pred,"Residual":abs(y_pred-Y_test)})



df
from sklearn import metrics

print("Mean Absolute Error (MAE):", metrics.mean_absolute_error(Y_test,y_pred))

print("Mean Squared Error (MSE):", metrics.mean_squared_error(Y_test ,y_pred))

print("Root Mean Squared Error (RMSE):", np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))

#Modelin tahmin başıan düşen ortalama hatası 7704.355038502031 diyebiliriz.Ve bu fazladır..

df.std()["Residual"]#verilen hata ortalamadan daha düşük olduğundan nispeten istikrarlı bir model diyebiliriz.

df.describe()["Residual"]#Kartiller üzerinden hesapladığımızda hiçbir üst ya da alt uç değere rastlamıyoruz.
#rss/tss eğer hesaplarsak 0.05 e yakın çıkıyor. Bu da Tss'nin 25 kat daha büyük olduğunu gösteriyor. Modelimiz çok başarılıdır. 

print("R Squared:", model.score(X_train, Y_train))