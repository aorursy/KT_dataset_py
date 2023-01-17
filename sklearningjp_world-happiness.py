import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df15 = pd.read_csv('../input/world-happiness/2015.csv')
df16 = pd.read_csv('../input/world-happiness/2016.csv')
df17 = pd.read_csv('../input/world-happiness/2017.csv')
df18 = pd.read_csv('../input/world-happiness/2018.csv')
df19 = pd.read_csv('../input/world-happiness/2019.csv')
df15 = pd.DataFrame(df15)
df16 = pd.DataFrame(df16)
df17 = pd.DataFrame(df17)
df18 = pd.DataFrame(df18)
df19 = pd.DataFrame(df19)
df15.head()
df15.columns
df16.head()
df17.head()
df18.head()
df19.head()
print(df15.columns)
print(df16.columns)
print(df17.columns)
x = df19['GDP per capita']
y = df19['Score']
value = df19['Freedom to make life choices'] #人生の選択権

plt.scatter(x, y, s=50, c=value, cmap='Blues')
 
# カラーバーを表示
plt.colorbar()
x = df19['GDP per capita']
y = df19['Score']
value = df19['Generosity']

plt.scatter(x, y, s=50, c=value, cmap='Blues')
 
# カラーバーを表示
plt.colorbar()
x = df19['GDP per capita']
y = df19['Score']
value = df19['Perceptions of corruption'] #腐敗認識指数

plt.scatter(x, y, s=50, c=value, cmap='Blues')
 
# カラーバーを表示
plt.colorbar()
x = df19['GDP per capita']
y = df19['Score']
value = df19['Healthy life expectancy'] #腐敗認識指数

plt.scatter(x, y, s=50, c=value, cmap='Blues')
 
# カラーバーを表示
plt.colorbar()
x = df19['GDP per capita']
y = df19['Score']
value = df19['Social support'] #社会保障

plt.scatter(x, y, s=50, c=value, cmap='Blues')
 
# カラーバーを表示
plt.colorbar()
import seaborn as sns
%matplotlib inline
sns.jointplot('GDP per capita', 'Score', data=df19)
sns.jointplot('GDP per capita', 'Score', data=df18)
sns.regplot('Economy..GDP.per.Capita.', 'Happiness.Score', data=df17)
sns.regplot('Economy (GDP per Capita)', 'Happiness Score', data=df16)
sns.regplot('Economy (GDP per Capita)', 'Happiness Score', data=df15)
df19.columns
sns.pairplot(df19, vars=['Score', 'GDP per capita', 'Social support', 'Healthy life expectancy',
       'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'])
df19_corr = df19.corr()
sns.heatmap(df19_corr, annot = True)
plt.show()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_list = ['GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
y_list = ['Score']
X = df19[X_list]
y = df19[y_list]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape)
lr = LinearRegression()
lr.fit(X, y)
lr.coef_
# 切片
lr.intercept_
lr.fit(X_train, y_train)
lr.score(X_train, y_train)
lr.coef_
y_pred = lr.predict(X_test)
y_pred
y_pred - y_test
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_pred, y_test)