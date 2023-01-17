# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
startups = pd.read_csv('../input/sp-startup/50_Startups.csv') 
df = startups
df.head()
df.info()
df.shape
df.isnull().sum()
df.corr()
corr=df.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values);
sns.scatterplot(x="R&D Spend", y="Profit", data= df);
sns.distplot(df["Profit"], bins=16, color="black");
sns.distplot(df["R&D Spend"], bins=16, color="blue");
sns.distplot(df["Marketing Spend"], bins=16, color="red");
sns.distplot(df["Administration"], bins=16, color="green");
#bağımlo değişken ile bağımsız değişken arasındaki ilişkinin gösterimi
sns.pairplot(df, x_vars=['R&D Spend','Administration','Marketing Spend'], y_vars='Profit', size=7, aspect=0.7)
df.describe().T
df["State"].unique()
df['State'] = pd.Categorical(df['State'])
dfDummies = pd.get_dummies(df['State'], prefix = 'State')
dfDummies
df = pd.concat([df, dfDummies], axis=1)
df.head()
df.drop(["State", "State_New York"], axis = 1, inplace = True)
X = df.drop("Profit", axis = 1)
y = df["Profit"]
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=99)
X_train.head()
X_test.head()
y_train.head()
y_test.head()
lm = LinearRegression()
model = lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
df_a = pd.DataFrame({'Gercek': y_test, 'Tahmin': y_pred})
df_a
MSE = mean_squared_error(y_test, y_pred)
MSE
RMSE = math.sqrt(MSE)
RMSE
MAE = mean_absolute_error(y_test, y_pred)
MAE
model.score(X_train, y_train)
model.score(X_test, y_test)
lm = sm.OLS(y, X)
model=lm.fit()
model.summary()
#analize başlamadan bazı varsayımlar kontrol edilmeli
#normallik testi
from numpy.random import seed
from numpy.random import randn
from scipy.stats import shapiro
seed(1)
for i in df:
    stat, p = shapiro(df[i])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
#doğrusallık testi
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
for i in df:
    qqplot(df[i], line='s')
    pyplot.show()
#artıkların dağılımı normal olmalı varsayımın test etme
residual = y_test - y_pred
sns.distplot(residual)
#hata terimlerinin ortalaması sıfır olmalı
np.mean(residual)
#bağımsız değişkenler arasında çoklu bağlantı olmamalı.
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
pd.DataFrame({'vif': vif[0:]}, index=X_train.columns).T
#sabit varyans olmalı
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,2.5))
_ = ax.scatter(y_pred, residual)
#filtreleme yöntemi için korelasyon kullanılabilir
cor = df.corr()
cor_target = abs(cor["Profit"])
#yüksek korelasyona sahip değişkenler seçilebilir
relevant_features = cor_target[cor_target>0.5]
relevant_features
print(df[["R&D Spend","Marketing Spend"]].corr())# iki değişken arasında yüksek korelasyon var ve beraber modelde yer almamalı
import statsmodels.formula.api as smf
df["rdS"]=df["R&D Spend"]
lm1 = smf.ols(formula='Profit ~ rdS ', data=df).fit()
lm1.params
lm1.rsquared
df_filter=df.filter(items=['R&D Spend', "Profit"])
df_filter
X = df_filter.drop("Profit", axis = 1)
y = df_filter["Profit"]
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=99)
lm = LinearRegression()
model = lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
df_a = pd.DataFrame({'Gercek': y_test, 'Tahmin': y_pred})
df_a
MSE = mean_squared_error(y_test, y_pred)
MSE
RMSE = math.sqrt(MSE)
RMSE
MAE = mean_absolute_error(y_test, y_pred)
MAE
model.score(X_train, y_train)
X_1 = sm.add_constant(X)
model = sm.OLS(y,X_1).fit()
degerler=model.pvalues
degerler.sort_values()
#Backward Elimination
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
