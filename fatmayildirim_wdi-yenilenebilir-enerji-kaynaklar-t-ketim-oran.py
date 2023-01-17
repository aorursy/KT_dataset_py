# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv("../input/yenilenebilirenerjikaynaklarituketimi.csv")

yenilenebilirenerjituketimi_metadata = pd.read_csv("../input/yenilenebilirenerjituketimi_metadata.csv")
data.head()
# Satir Sayisi

print("Satır Sayısı:\n",data.shape[0:])



# Sutun Adlari

print("Sütun Adlari:\n",data.columns.tolist())



# Veri Tipleri

print("Veri Tipleri:\n",data.dtypes)

# Eksik veri sayıları ve veri setindeki oranları 

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(8,8))

sns.heatmap(pd.isnull(data.T), cbar=False)



pd.concat([data.isnull().sum(), 100 * data.isnull().sum()/len(data)], 

              axis=1).rename(columns={0:'Missing Records', 1:'Percentage (%)'})
# 1998 yılı haricindekiler kategorik değişkenden sürekli değişkene dönüştürüldü.

dt=['YRbir','YRiki','YRuc','YRdort','YRbes','YRalti','YRyedi','YRdokuz','YRon','YRonbir','YRoniki','YRonuc','YRondort','YRonbes','YRonalti','YRonyedi','YRonsekiz','YRondokuz','YRyirmi','YRyirmibir','YRyirmiiki','YRyirmiuc','YRyirmidort', 'YRyirmibes','YRyirmialti' ]

for i in  dt:

  data[i] = pd.to_numeric(data[i], errors = 'coerce')

data.info()
data['YRbir']
data_a=data.copy()

y = (data_a['Country Name'] == 'Turkey').astype(int)

fields = list(data_a.columns[:-1])  # everything except "country name"

correlations = data_a[fields].corrwith(y)

correlations.sort_values(inplace=True)

correlations

ax = correlations.plot(kind='bar')

ax.set(ylim=[-1, 1], ylabel='turkey correlation');
# Sürekli değişken sütunlarındaki boş alanlar ortalama değerler ile dolduruldu.

cols = ['YRbir','YRiki','YRuc','YRdort','YRbes','YRalti','YRyedi' ,'YRdokuz','YRon','YRonbir','YRoniki','YRonuc','YRondort','YRonbes','YRonalti','YRonyedi','YRonsekiz','YRondokuz','YRyirmi','YRyirmibir','YRyirmiiki','YRyirmiuc','YRyirmidort', 'YRyirmibes','YRyirmialti']

for i in cols:

   data[i].fillna(data[i].mean(),inplace=True)
#Yalnızca kategorik değişkenlerde ve YRsekizde boş alanlar kalmıştır. 

for i in data:

  df=data[i].isnull().values.sum()

  print(df)
# Seçilmiş olan yıllarla yeni bir dataframe oluşturuldu.

df1=pd.Series(data['Country Name'],name="CountryName")

df2=pd.Series(data['YRbir'],name="YRbir")

df3=pd.Series(data['YRalti'],name="YRalti")

df4=pd.Series(data['YRonbir'],name="YRonbir")

df5=pd.Series(data['YRonalti'],name="YRonalti")

df6=pd.Series(data['YRyirmibir'],name="YRyirmibir")

df7=pd.Series(data['YRyirmialti'],name="YRyirmialti")

df=pd.concat([df1, df2,df3, df4,df5, df6,df7], axis=1)
df.describe().T
# Aykırı değerleri gözlemleyebilmek için box plot kullanıldı

plt.figure()

df.boxplot(column=['YRbir','YRalti','YRonbir','YRonalti','YRyirmibir','YRyirmialti'])



fig,axs=plt.subplots(2,3) 

axs[0, 0].boxplot(df['YRbir'])

axs[0, 0].set_title('YRbir')



axs[0, 1].boxplot(df['YRalti'])

axs[0, 1].set_title('YRalti')



axs[0, 2].boxplot(df['YRonbir'])

axs[0, 2].set_title('YRonbir')



axs[1, 0].boxplot(df['YRonalti'])

axs[1, 0].set_title('YRonalti')



axs[1, 1].boxplot(df['YRyirmibir'])

axs[1, 1].set_title('YRyirmibir')



axs[1, 2].boxplot(df['YRyirmialti'])

axs[1, 2].set_title('YRyirmialti')
# Histogram grafiği

from matplotlib import pyplot

df.hist()

pyplot.show()
# Scatter Plot Matrix

from pandas.plotting import scatter_matrix

scatter_matrix(df)

pyplot.show()

import numpy as np

from sklearn    import metrics, svm

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn import  linear_model

array = df.values

X = array[:,1:6]

y = array[:,6]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.25, random_state=1, shuffle=True)

print("Dataframe boyutu: ",df.shape)

print("Eğitim verisi boyutu: ",X_train.shape, Y_train.shape)

print("Test verisi boyutu: ",X_validation.shape, Y_validation.shape)

# type error için target typesı "Label Encoder" ile  multiclassa çevirdim.(Target=Y_train)

from sklearn import preprocessing

from sklearn import utils



lab_enc = preprocessing.LabelEncoder()

encoded = lab_enc.fit_transform(y)

print(utils.multiclass.type_of_target(y))

print(utils.multiclass.type_of_target(Y_train.astype('int')))

print(utils.multiclass.type_of_target(encoded))



lab_enc = preprocessing.LabelEncoder()

Y_train = lab_enc.fit_transform(Y_train)

print(utils.multiclass.type_of_target(Y_train))
# Modeller

models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))

# modellerin sırasıyla değerlendirilmeleri

results = []

names = []

for name, model in models:

	kfold = StratifiedKFold(n_splits=10, random_state=1)

	cv_results = cross_val_score(model, X, encoded, cv=kfold, scoring='accuracy')

	results.append(cv_results)

	names.append(name)

	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Algoritmalrın boxplot üzerinde karşılaştırılıp aykırı değer tespiti yapılması

pyplot.boxplot(results, labels=names)

pyplot.title('Algorithm Comparison')

pyplot.show()
# Her bir modelin doğruluk değeri ,sınıflandırma raporu , karışıklık matrisi ve MSE(Ortalama Kare Hata Regresyon Oranı) değerlerini hesaplamak için import edildi.

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import mean_squared_error
# Lineer Regresyon

print("\nLineer Regresyon")

lm = linear_model.LinearRegression()

model = lm.fit(X_train, Y_train)

y_true1 , y_pred1 =Y_validation,lm.predict(X_validation)

print("\nTahmin değerleri: ",y_pred1)

plt.scatter(y_true1, y_pred1,c='orange')

plt.scatter(y_true1, Y_validation,c='green')

plt.xlabel("True Values")

plt.ylabel("Predictions")
#Lineer Regresyon

#predictions multiclass olduğundan y_validation da multiclassa dönüştürüldü

encoded_v = lab_enc.fit_transform(y_true1)

utils.multiclass.type_of_target(y_true1.astype('int'))

ypred1= lab_enc.fit_transform(y_pred1)

utils.multiclass.type_of_target(ypred1.astype('int'))

conf=confusion_matrix(encoded_v, ypred1)

print("\nConfusion matrix :\n",conf)

sns.heatmap(conf, cmap="Blues")



#Lineer Regresyon

print("Accuracy score(Doğruluk değeri):\n",accuracy_score(encoded_v, ypred1))

print("\nClassification Report:\n",classification_report(encoded_v, ypred1))

print("MSE:",mean_squared_error(encoded_v, ypred1))
# SVR(Support Vector Regressions)

print("SVR(Support Vector Regressions)")

clf = svm.SVR(gamma="auto")

# modelimizi eğitim verilerimiz ve buna karşılık gelen Y_train(target ) değerleri ile eğittik

clf.fit(X_train, Y_train)

# test değerlerimize karşılık gelecek olan tahmin değerlerimizi oluşturduk

y_true2 , y_pred2 =Y_validation,clf.predict(X_validation)

print("\nTahmin değerleri: ",y_pred2)

plt.scatter(y_true2, y_pred2,c='black')

plt.scatter(y_true2, Y_validation,c='green')

plt.xlabel("True Values")

plt.ylabel("Predictions")

#SVR

#predictions multiclass olduğundan y_validation da multiclassa dönüştürüldü

encoded_v1 = lab_enc.fit_transform(y_true2)

utils.multiclass.type_of_target(y_true2.astype('int'))

ypred2= lab_enc.fit_transform(y_pred2)

utils.multiclass.type_of_target(ypred2.astype('int'))

conf=confusion_matrix(encoded_v1, ypred2)

print("\nConfusion matrix :\n",conf)

sns.heatmap(conf, cmap="Blues")



print("Accuracy score(Doğruluk değeri):\n",accuracy_score(encoded_v1, ypred2))

print("\nClassification Report:\n",classification_report(encoded_v1, ypred2))

print("MSE:",mean_squared_error(encoded_v1, ypred2))
# SVC

print("SVC")

clf = SVC(gamma="auto")

clf.fit(X_train, Y_train)

y_true3 , y_pred3 =Y_validation,clf.predict(X_validation)

print("\nTahmin değerleri: ",y_pred3)

plt.scatter(y_true3, y_pred3,c='yellow')

plt.scatter(y_true3, Y_validation,c='green')

plt.xlabel("True Values")

plt.ylabel("Predictions")
#SVC

#predictions multiclass olduğundan y_validation da multiclassa dönüştürüldü

encoded_v2 = lab_enc.fit_transform(y_true3)

utils.multiclass.type_of_target(y_true3.astype('int'))

ypred3= lab_enc.fit_transform(y_pred3)

utils.multiclass.type_of_target(ypred3.astype('int'))

conf=confusion_matrix(encoded_v2, ypred3)

print("\nConfusion matrix :\n",conf)

sns.heatmap(conf, cmap="Blues")





print("Accuracy score(Doğruluk değeri):\n",accuracy_score(encoded_v2, ypred3))

print("\nClassification Report:\n",classification_report(encoded_v2, ypred3))

print("MSE:",mean_squared_error(encoded_v2, ypred3))
# GaussianNB

print("GaussianNB")

clf = GaussianNB()

clf.fit(X_train, Y_train)

y_true4 , y_pred4=Y_validation,clf.predict(X_validation)

print("\nTahmin değerleri: ",y_pred4)

plt.scatter(y_true4, y_pred4,c='grey')

plt.scatter(y_true4, Y_validation,c='green')

plt.xlabel("True Values")

plt.ylabel("Predictions")

# GaussianNB

#predictions multiclass olduğundan y_validation da multiclassa dönüştürüldü

encoded_v3 = lab_enc.fit_transform(y_true4)

utils.multiclass.type_of_target(y_true4.astype('int'))

ypred4= lab_enc.fit_transform(y_pred4)

utils.multiclass.type_of_target(ypred4.astype('int'))

conf=confusion_matrix(encoded_v3, ypred4)

print("\nConfusion matrix :\n",conf)

sns.heatmap(conf, cmap="Blues")





print("Accuracy score(Doğruluk değeri):\n",accuracy_score(encoded_v3, ypred4))

print("\nClassification Report:\n",classification_report(encoded_v3, ypred4))

print("MSE:",mean_squared_error(encoded_v3, ypred4))
# Decision Tree Classifier

print("Decision Tree Classifier")

clf = DecisionTreeClassifier()

clf.fit(X_train, Y_train)

y_true5 , y_pred5=Y_validation,clf.predict(X_validation)

print("\nTahmin değerleri: ",y_pred5)

plt.scatter(y_true5, y_pred5,c='brown')

plt.scatter(y_true5, Y_validation,c='green')

plt.xlabel("True Values")

plt.ylabel("Predictions")
# Decision Tree Classifier

#predictions multiclass olduğundan y_validation da multiclassa dönüştürüldü

encoded_v4 = lab_enc.fit_transform(y_true5)

utils.multiclass.type_of_target(y_true5.astype('int'))

ypred5= lab_enc.fit_transform(y_pred5)

utils.multiclass.type_of_target(ypred5.astype('int'))

conf=confusion_matrix(encoded_v4, ypred5)

print("\nConfusion matrix :\n",conf)

sns.heatmap(conf, cmap="Blues")





print("Accuracy score(Doğruluk değeri):\n",accuracy_score(encoded_v4, ypred5))

print("\nClassification Report:\n",classification_report(encoded_v4, ypred5))

print("MSE:",mean_squared_error(encoded_v4, ypred5))
# Logistic Regresyon

from sklearn.linear_model import LogisticRegression

print("Logistic Regression")

clf = LogisticRegression(multi_class="auto")

clf.fit(X_train, Y_train)

y_true6 , y_pred6=Y_validation,clf.predict(X_validation)

print("\nTahmin değerleri: ",y_pred6)

plt.scatter(y_true6, y_pred6,c='purple')

plt.scatter(y_true6, Y_validation,c='green')

plt.xlabel("True Values")

plt.ylabel("Predictions")

# Logistic Regresyon

#predictions multiclass olduğundan y_validation da multiclassa dönüştürüldü

encoded_v5 = lab_enc.fit_transform(y_true6)

utils.multiclass.type_of_target(y_true6.astype('int'))

ypred6= lab_enc.fit_transform(y_pred6)

utils.multiclass.type_of_target(ypred6.astype('int'))

conf=confusion_matrix(encoded_v5, ypred6)

print("\nConfusion matrix :\n",conf)

sns.heatmap(conf, cmap="Blues")





print("Accuracy score(Doğruluk değeri):\n",accuracy_score(encoded_v5, ypred6))

print("\nClassification Report:\n",classification_report(encoded_v5, ypred6))

print("MSE:",mean_squared_error(encoded_v5, ypred6))
# KNeighborsClassifier

print("KNeighbors Classifier")

clf = KNeighborsClassifier()

clf.fit(X_train, Y_train)

y_true7 , y_pred7=Y_validation,clf.predict(X_validation)

print("\nTahmin değerleri: ",y_pred7)

plt.scatter(y_true7, y_pred7,c='blue')

plt.scatter(y_true7, Y_validation,c='green')

plt.xlabel("True Values")

plt.ylabel("Predictions")
# KNeighborsClassifier

#predictions multiclass olduğundan y_validation da multiclassa dönüştürüldü

encoded_v6 = lab_enc.fit_transform(y_true7)

utils.multiclass.type_of_target(y_true7.astype('int'))

ypred7= lab_enc.fit_transform(y_pred7)

utils.multiclass.type_of_target(ypred7.astype('int'))

conf=confusion_matrix(encoded_v6, ypred7)

print("\nConfusion matrix :\n",conf)

sns.heatmap(conf, cmap="Blues")





print("Accuracy score(Doğruluk değeri):\n",accuracy_score(encoded_v6, ypred7))

print("\nClassification Report:\n",classification_report(encoded_v6, ypred7))

print("MSE:",mean_squared_error(encoded_v6, ypred7))
# Linear Discriminant Analysis

print("Linear Discriminant Analysis")

clf = LinearDiscriminantAnalysis()

clf.fit(X_train, Y_train)

y_true8 , y_pred8=Y_validation,clf.predict(X_validation)

print("\nTahmin değerleri: ",y_pred8)

plt.scatter(y_true8, y_pred8,c='red')

plt.scatter(y_true8, Y_validation,c='green')

plt.xlabel("True Values")

plt.ylabel("Predictions")



# Linear Discriminant Analysis

#predictions multiclass olduğundan y_validation da multiclassa dönüştürüldü

encoded_v7 = lab_enc.fit_transform(y_true8)

utils.multiclass.type_of_target(y_true8.astype('int'))

ypred8= lab_enc.fit_transform(y_pred8)

utils.multiclass.type_of_target(ypred8.astype('int'))

conf=confusion_matrix(encoded_v7, ypred8)

print("\nConfusion matrix :\n",conf)

sns.heatmap(conf, cmap="Blues")





print("Accuracy score(Doğruluk değeri):\n",accuracy_score(encoded_v7, ypred8))

print("\nClassification Report:\n",classification_report(encoded_v7, ypred8))

print("MSE:",mean_squared_error(encoded_v7, ypred8))