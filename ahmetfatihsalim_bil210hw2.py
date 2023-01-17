import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns # kütüphaneleri ekledik
df = pd.read_csv("../input/startups/50_Startups.csv").copy()
df.head()
df.info()
df.shape   # (50 gözlem,5 öznitelikten oluşuyor)
df.isnull().sum()#görünürde boş gözlem yok
df.corr()# en güçlü ilişki Profit ve R&D Spend arasında iken en zayıf ilişki Marketing Spend ve Administration arasında
sns.heatmap(df.corr())
#Isı haritasında Profit ve R&D Spend arasında mükemmel bir ilişki olduğu gözlemlenebilir
#yönetim ile pazarlama harcamaları arasında bir ilişki yok demek yanlış olmaz
sns.scatterplot(x="R&D Spend",y="Profit",data = df)# aralarındaki ilişkinin yüksek olduğundan R&D Spend artarken profitte artıyor.
df.hist()
df.describe().T#market ve araştırma geliştirme harcaması olmayan şirketler görünüyor
# harcamalar , standart sapmaya göre oldukça değişken
df["State"].unique()#State'e ait benzersiz değerler :New York,California,Florida
df["State"] = pd.Categorical(df["State"])# state özniteliğine göre kategorik çıkarımda bulundum
stateDummies = pd.get_dummies(df["State"], prefix = "State")
df = pd.concat([df,stateDummies],axis = 1)
df.head()
df.drop(["State","State_Florida"],axis = 1, inplace = True)
df.head()
y = df['Profit']#bağımlı değişken: kar
X = df.drop(['Profit'], axis=1)#bağımsız değişkenler
X
y
from sklearn.model_selection import train_test_split #verileri eğitime, teste sokmak amacı ile eklediğimiz kütüphane
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)#verilerimiz 0.25 i test edilecek biçimde oranlandı. 
X_train
X_test
y_train
y_test
from sklearn.linear_model import LinearRegression 
model=LinearRegression() 
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
df_gozlem = pd.DataFrame({"Gercek deger" : y_test, "Tahmini deger" : y_pred})

df_gozlem
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, precision_score
from sklearn import metrics
print("mean squared error(MSE)" , metrics.mean_squared_error(y_test,y_pred))#buradaki değerler 1 e ne kadar yakınsa model bizim için o kadar iyi demektir
print("root mean squared error(RMSE)" , np.sqrt(metrics.mean_squared_error(y_test,y_pred)))#ne kadar düşükse o kadar başarılıdır.
import statsmodels.api as stat
stmodel = stat.OLS(y_train, X_train).fit()
stmodel.summary()