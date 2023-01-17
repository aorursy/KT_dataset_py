# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
movie=pd.read_csv("../input/imdb-extensive-dataset/IMDb movies.csv")
movie.head()
movie.shape
# Satir Sayisi
print("Satır Sayısı:\n",movie.shape[0:])

# Sutun Adlari
print("Sütun Adlari:\n",movie.columns.tolist())

# Veri Tipleri
print("Veri Tipleri:\n",movie.dtypes)
# Eksik veri sayıları ve veri setindeki oranları 
import matplotlib.pyplot as plt
import seaborn as sns
pd.concat([movie.isnull().sum(), 100 * movie.isnull().sum()/len(movie)], 
              axis=1).rename(columns={0:'Missing Records', 1:'Percentage (%)'})
#Dram türü için korelasyon grafiği
drama=movie.copy()
drama_values = (drama['genre'] == 'Drama').astype(int)
fields = list(drama.columns[:-4])
correlations = drama[fields].corrwith(drama_values)
correlations.sort_values(inplace=True)
correlations
ax = correlations.plot(kind='bar')
ax.set(ylim=[-1, 1], ylabel='Drama Correlation');
#Suç türü için korelasyon grafiği
crime=movie.copy()
crime_values = (drama['genre'] == 'Crime').astype(int)
fields = list(drama.columns[:-4])  
correlations = drama[fields].corrwith(crime_values)
correlations.sort_values(inplace=True)
correlations
ax = correlations.plot(kind='bar')
ax.set(ylim=[-1, 1], ylabel='Crime Correlation');
# Veri seti içerisinden belli alanlar seçilerek yeni bir veriseti oluşturuldu.

df1=pd.Series(movie['duration'],name="duration")
df2=pd.Series(movie['votes'],name="votes")
df_movie=pd.concat([df1,df2], axis=1)
df_movie.describe().T
plt.figure()
df_movie.boxplot(column=['duration','votes'])

fig,axs=plt.subplots(2,2) 
axs[0, 0].boxplot(df_movie['duration'])
axs[0, 0].set_title('duration')

axs[0, 1].boxplot(df_movie['votes'])
axs[0, 1].set_title('votes')




# Histogram grafiği
from matplotlib import pyplot
df_movie.hist()
pyplot.show()
# Scatter Plot Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(df_movie)
pyplot.show()
#Amerikada izlenen filmler için korelasyon grafiği
us=movie.copy()
y = (us['country'] == 'USA').astype(int)
fields = list(us.columns[:-1])  # everything except "country name"
correlations = us[fields].corrwith(y)
correlations.sort_values(inplace=True)
correlations

ax = correlations.plot(kind='bar')
ax.set(ylim=[-1, 1], ylabel='USA Correlation');
plt.figure()
movie.boxplot(column=['year','duration','avg_vote','metascore','votes','reviews_from_users'])

fig,axs=plt.subplots(2,3) 
axs[0, 0].boxplot(movie['year'])
axs[0, 0].set_title('Film Yılı')

axs[0, 1].boxplot(movie['duration'])
axs[0, 1].set_title('Film Süresi')

axs[0, 2].boxplot(movie['avg_vote'])
axs[0, 2].set_title('Film Hakkında Yapılan Ortalama Oy Sayısı')

axs[1, 0].boxplot(movie['metascore'])
axs[1, 0].set_title('Metascore')

axs[1, 1].boxplot(movie['votes'])
axs[1, 1].set_title('Film Hakkında Yapılan Oylama Sayısı')

axs[1, 2].boxplot(movie['reviews_from_users'])
axs[1, 2].set_title('Film Hakkında Kullanıcı Yorumları')

# Film süresine göre oylamanın nasıl olduğu hakkında bilgi almak için veri eğitildi.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X = df_movie
y = df_movie['votes']

mms = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, random_state=0)
X_train = mms.fit_transform(X_train) 
X_test= mms.fit_transform(X_test)

print("Dataframe boyutu: ",df_movie.shape)
print("Eğitim verisi boyutu: ",X_train.shape, y_train.shape)
print("Test verisi boyutu: ",X_test.shape,y_test.shape)
# type error için target typesı "Label Encoder" ile  multiclassa çevirdim.(Target=Y_train)
from sklearn import preprocessing
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y)
print(utils.multiclass.type_of_target(y))
print(utils.multiclass.type_of_target(y_train.astype('int')))
print(utils.multiclass.type_of_target(encoded))

lab_enc = preprocessing.LabelEncoder()
Y_train = lab_enc.fit_transform(y_train)
import numpy as np
from sklearn    import metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import  linear_model
# Her bir modelin doğruluk değeri ,sınıflandırma raporu , karışıklık matrisi ve MSE(Ortalama Kare Hata Regresyon Oranı) değerlerini hesaplamak için import edildi.
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
# Lineer Regresyon
print("\nLineer Regresyon")
lm = linear_model.LinearRegression()
model = lm.fit(X_train, Y_train)
y_true1 , y_pred1 =y_test,lm.predict(X_test)
print("\nTahmin değerleri: ",y_pred1)
plt.scatter(y_true1, y_pred1,c='blue')
plt.scatter(y_true1, y_test,c='pink')
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
print("Accuracy score(Doğruluk değeri):\n",accuracy_score(encoded_v, ypred1))
print("\nClassification Report:\n",classification_report(encoded_v, ypred1))
print("MSE:",mean_squared_error(encoded_v, ypred1))
# Decision Tree Classifier
print("Decision Tree Classifier")
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
y_true5 , y_pred5=y_test,clf.predict(X_test)
print("\nTahmin değerleri: ",y_pred5)
plt.scatter(y_true5, y_pred5,c='orange')
plt.scatter(y_true5, y_test,c='purple')
plt.xlabel("True Values")
plt.ylabel("Predictions")
#predictions multiclass olduğundan y_validation da multiclassa dönüştürüldü
encoded_v4 = lab_enc.fit_transform(y_true5)
utils.multiclass.type_of_target(y_true5.astype('int'))
ypred5= lab_enc.fit_transform(y_pred5)
utils.multiclass.type_of_target(ypred5.astype('int'))
conf=confusion_matrix(encoded_v4, ypred5)
print("\nConfusion matrix :\n",conf)
print("Accuracy score(Doğruluk değeri):\n",accuracy_score(encoded_v4, ypred5))
print("\nClassification Report:\n",classification_report(encoded_v4, ypred5))
print("MSE:",mean_squared_error(encoded_v4, ypred5))
# GaussianNB
print("GaussianNB")
clf = GaussianNB()
clf.fit(X_train, Y_train)
y_true3 , y_pred3=y_test,clf.predict(X_test)
print("\nTahmin değerleri: ",y_pred3)
plt.scatter(y_true3, y_pred3,c='pink')
plt.scatter(y_true3, y_test,c='orange')
plt.xlabel("True Values")
plt.ylabel("Predictions")
# GaussianNB
#predictions multiclass olduğundan y_validation da multiclassa dönüştürüldü
encoded_v2 = lab_enc.fit_transform(y_true3)
utils.multiclass.type_of_target(y_true3.astype('int'))
ypred3= lab_enc.fit_transform(y_pred3)
utils.multiclass.type_of_target(ypred3.astype('int'))
conf=confusion_matrix(encoded_v2, ypred3)
print("\nConfusion matrix :\n",conf)
print("Accuracy score(Doğruluk değeri):\n",accuracy_score(encoded_v2, ypred3))
print("\nClassification Report:\n",classification_report(encoded_v2, ypred3))
print("MSE:",mean_squared_error(encoded_v2, ypred3))