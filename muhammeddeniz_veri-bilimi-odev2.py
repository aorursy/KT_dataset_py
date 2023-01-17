import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

startups = pd.read_csv("../input/50_Startups.csv")
df = startups.copy();

# kendi dataset üzerinde işlem yapmak veri güvenliği için sıkıntılı olabilir Bu yüzden veri seti kopyası üzerinden 
# devam etmek daha iyi olur.
df.head()
df.info()
df.shape

df.isna()
df.isna().sum()

#isnull() fonksiyonu da kullanılabilir.
corr = df.corr()
corr
sns.heatmap(corr, xticklabels = corr.columns.values, yticklabels = corr.columns.values)

# veriler arasındaki korelasyon ilişkisini daha iyi görebilmek için heatmap() gerekli.
sns.set(rc={'figure.figsize':(10,8)});
sns.scatterplot(df['R&D Spend'], df["Profit"])

# korelasyon çizgisini burada çizmiyoruz fakat yine de belirli bir sınırı bize gösteriyor.

sns.regplot(df["Profit"], df["R&D Spend"], ci=None);

# Burda çizgimizi çizip sapmaları daha net görebiliriz.
df.describe().T
df['State'].unique()
pd.get_dummies(df["State"])["California"].head()
pd.get_dummies(df["State"])["New York"].head()
pd.get_dummies(df["State"])["Florida"].head()
pd.get_dummies(df["State"]).sum()
deneme = df["State"].copy()

deneme['Florida'] = pd.get_dummies(df["State"])["Florida"]
deneme['New York'] = pd.get_dummies(df["State"])["New York"]
deneme['California'] = pd.get_dummies(df["State"])["California"]
deneme
y = df["Profit"]
X = df["R&D Spend"]

y.head()
# Bağımlı değişken

X.head()
# Bağımsız değişken
X_train,X_test, y_train, y_test = train_test_split(X,y);
X_train
X_test
y_train
y_test
from sklearn.linear_model import LinearRegression

linear_regresyon = LinearRegression()

df.head(2)
X = df.drop("Profit", axis=1)
X = X.drop("State", axis=1)
X = X.drop("Marketing Spend", axis=1)
X = X.drop("Administration", axis=1)

# değerlerden sadece bağımsız ve bağımlı değişkenleri alıyoruz

y = df["Profit"]
X
y
linear_regresyon.fit(X, y)

# modelimizi deniyoruz
# birinci satır için bir deneme yapalım
deneme1 = linear_regresyon.predict([[165349.20]])
print("birinci deneme: ", deneme1)

# bir tane de rastgele değişkenlere tahmin yaptıralım
deneme2 = linear_regresyon.predict([[2000]])
print("ikinci deneme: ", deneme2)

# denemelerimizden  gerçek değer olanını tablodan bakıp sapma oranını belirli bir değer için gözlemleyebiliriz.
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import mean_absolute_error

MSE = mean_squared_error(df["Profit"], df["R&D Spend"])
print("MSE: ", MSE)

RMSE = math.sqrt(MSE)
print("RMSE: ", RMSE)

MAE = mean_absolute_error(df["Profit"], df["R&D Spend"])
print("MAE: ", MAE)
linear_regresyon.score(X,y)
import statsmodels.api as sm
stmodel = sm.OLS(y, X).fit()
stmodel.summary()
