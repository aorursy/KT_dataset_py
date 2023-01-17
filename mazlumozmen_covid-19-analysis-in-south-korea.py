import random
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import folium

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()   

import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data=pd.read_csv('../input/coronavirusdataset/Case.csv')

data.head()
data2=pd.read_csv('../input/coronavirusdataset/Time.csv')

data2.head()
print("Satır sayısı",data.shape[0])

print("Sütun Adları",data.columns.tolist())

print("Veri Tiplerı",data.dtypes)

print("Satır sayısı",data2.shape[0])

print("Sütun Adları",data2.columns.tolist())

print("Veri Tiplerı",data2.dtypes)

data.groupby('province').mean()
data2.mean()
data.info()
data2.info()
data.tail()
print(data.groupby('province').size())
data.sort_values(by=['confirmed'], ascending=False, inplace = True)
fig = px.bar(data, x="province", y="confirmed", title='Şehirlerdeki Vaka Sayıları')
fig.update_layout(barmode='group')
fig.show()
fig = px.pie(data, values='confirmed', names='province', title='Şehirlerindeki vaka sayıları', hover_data=['confirmed'])
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
data2.head(-10).T
from datetime import date


data2['date'] = pd.to_datetime(data2['date'])
data2['date'] = data2['date'].dt.date

confirmed=data2["confirmed"].values
gun_sayisi=len(data2[data2['date']>date(2020,1,20)])
x=np.arange(0 , gun_sayisi)
gunluk_vaka=[]
for n in x:
    gunluk_vaka.append(confirmed[x-n]-confirmed[x-n-1])
    if n == 0:
        gunluk_vaka[0][0]=1
        break
print(gunluk_vaka)
x=x.tolist()
gunluk_vaka=gunluk_vaka[0].tolist()
x=pd.Series(x)
gunluk_vaka=pd.Series(gunluk_vaka)
print(x.shape)
print(gunluk_vaka.shape)
gun_sayisi=len(data2[data2['date']>date(2020,1,20)])

x=x.values.reshape(-1,1)
y=gunluk_vaka.values.reshape(-1,1)
print(x.shape)
print(y.shape)
plt.figure(figsize=(16, 9))
plt.scatter(x,y)
plt.xlabel("Gün Sayısı")
plt.ylabel("Vaka Sayısı")
plt.title("Günlük Vaka Sayısı")
plt.show()
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
Poly_reg=PolynomialFeatures(degree=8)  
x_poly=Poly_reg.fit_transform(x)
Lin_reg = LinearRegression()
Lin_reg.fit(x_poly,y)
print(x_poly[:8])
plt.figure(figsize=(16, 9))
plt.scatter(x,y)
plt.xlabel("Gün Sayısı")
plt.ylabel("Vaka Sayısı")
plt.title("Regresyon Modeli")
y_pred=Lin_reg.predict(x_poly)
plt.plot(x,y_pred,color="green",label="Polinom Linner Regresyon Model")
plt.legend()
plt.show()
print(Lin_reg.intercept_)
print(Lin_reg.coef_)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
data3=data2

del data3['date']
array = data3.values
X = array[:,0:5]
y = array[:,5]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.60, random_state=1)
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
plt.boxplot(results, labels=names)
plt.title('Algoritma Karşılaştırması')
plt.show()
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
