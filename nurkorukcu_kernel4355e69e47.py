# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
data = pd.read_csv("/kaggle/input/coronavirus-cases-in-india/Covid cases in India.csv") # Veri seti yolu
data.head(10) # Verilerin görünümü
import matplotlib.pyplot as plt
%matplotlib inline
ax = plt.axes()
ax.scatter(data["S. No."], data.Active)

# Eksenleri isimlendirme
ax.set(xlabel='Günler',
       ylabel='Vaka Sayısı',
       title='Günlük pozitif vaka sayısı');
data.info() # Data set hakkında bilgi
data.describe() # İstatistiksel veriler
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(data.corr()); # Isı haritası
import seaborn as seabornInstance
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(data['Deaths']) # ölüm sayılarının dağılımı
df = data.dropna(how='any',axis=0)
df=data.filter(['Total Confirmed cases','Deaths','Active' , "Cured/Discharged/Migrated"]) # İstediğimiz alanları df diye ayrı bir değişkene alıyoruz.
df["Total Confirmed cases"]=pd.to_numeric(df["Total Confirmed cases"])
df["Deaths"]=pd.to_numeric(df["Deaths"])
df["Active"]=pd.to_numeric(df["Active"])
df["Cured/Discharged/Migrated"]=pd.to_numeric(df["Cured/Discharged/Migrated"])
df.head(10)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score 
X = df.drop(['Total Confirmed cases'], axis=1)
y = df['Total Confirmed cases']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train,y_train) # Model kuruldu
r_sq = model.score(X_train, y_train)
print('coefficient of determination:', r_sq)
print(model.intercept_)
print("R score: {0}".format(round(model.score(X_train, y_train),2)))
from sklearn.metrics import accuracy_score
predictions = model.predict(X_test)
df1 = pd.DataFrame({'Gerçek': y_test, 'Tahmin edilen': predictions.flatten()})
df1
sns.pairplot(df, x_vars=['Deaths','Active' , "Cured/Discharged/Migrated"], y_vars='Total Confirmed cases', size=8, aspect=0.9, kind='reg')
X = df.drop(['Total Confirmed cases'], axis=1)
y = df['Total Confirmed cases']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
Knn = KNeighborsClassifier(n_neighbors=5,weights= 'uniform')
Knn.fit(X, y)
y_pred = Knn.predict(X)
predictions = Knn.predict(X_test)
predictions
df2 = pd.DataFrame({'Gerçek': y_test, 'Tahmin edilen': predictions.flatten()})
df2
