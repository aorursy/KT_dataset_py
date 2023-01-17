# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for visualization
%matplotlib inline
import seaborn as sns #for visualization


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data =pd.read_csv('../input/Social_Network_Ads.csv')#pandas kütüphanesini kullanarak dizini atadık
data
#Veri Keşfi ve Görselleştirme
data.describe().T#istatistikler
data.info()#bellek kullanımı ve veri türleri
data.head() #ilk 5 satır
data.shape #satır ve sütun sayısı
data.tail()#son 5 satır
#Histogram grafiği incelemesi
data.hist(figsize=(16,16))
data.sample(6) #rastgele 6 satır
data.corr()
#Korelasyon Gösterim
f,ax = plt.subplots(figsize = (12,9))
sns.heatmap(data.corr(), annot = True, linewidths =.5, fmt = '.2f', ax=ax)
plt.show() 
#plotting
data.plot(x='Age', y='Purchased', style='o')  
plt.title('Age-Purchased')  
plt.xlabel('Age')  
plt.ylabel('Purchased')  
plt.show() 

data.plot(x='EstimatedSalary', y='Purchased', style='+')  
plt.title('EstimatedSalary-Purchased')  
plt.xlabel('EstimatedSalary')  
plt.ylabel('Purchased')  
plt.show()

#Ön İşleme
#Eksik Değer Doldurma
#Null olan öznitelikleri buluyoruz
data.isnull().sum()
#Null olan özniteliklere sahip, toplam kayıt sayısını buluyoruz
data.isnull().sum().sum()
#Eksik değer tablosu
def eksik_deger_tablosu(data): 
    mis_val = data.isnull().sum()
    mis_val_percent = 100 * data.isnull().sum()/len(data)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return mis_val_table_ren_columns
eksik_deger_tablosu(data)
#Bütün kolonlardaki Null değerleri 'boş' değeri ile doldur
data['User ID'] = data['User ID'].fillna('boş')
data['Gender'] = data['Gender'].fillna('boş')
data['Age'] = data['Age'].fillna('boş')
data['EstimatedSalary'] = data['EstimatedSalary'].fillna('boş')
data['Purchased'] = data['Purchased'].fillna('boş')

data
#Aykırı(Uç) Değer Tespiti
Global_Salesfig, axs = plt.subplots(ncols = 2, figsize=(15, 4))
sns.boxplot(data.Age, ax=axs[0], orient = 'h')
sns.boxplot(data.EstimatedSalary, ax=axs[1], orient = 'h', showfliers=False)
#Age alanındaki yaş bilgisini kullanarak 'Birthyear' isimli yeni bir öznitelik oluşturuyoruz
def dogum_yili(age):
    return (2018-age)
data['Birthyear'] = data['Age'].apply(dogum_yili)
data
#Veri Normalleştirme
from sklearn import preprocessing

#EstimatedSalary özniteliğini normalleştirmek istiyoruz
x = data[['EstimatedSalary']].values.astype(float)

#Normalleştirme için MinMax normalleştirme yöntemini kullanıyoruz.
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data['EstimatedSalary2'] = pd.DataFrame(x_scaled)
data
#Model Eğitimi
# Veri setimizi okuyoruz
data=pd.read_csv('../input/Social_Network_Ads.csv')
X = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values
# Veri setini test ve eğitim olarak 2'ye ayırıyoruz.
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Özellik ölçekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# eğitim setine Naive Bayes uyguluyoruz 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Test veri setini kullanarak sonuçları tahmin ediyoruz
y_pred = classifier.predict(X_test)
# Confusion matrisimizi oluşturuyoruz.
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
# Eğitim sonuçları gözlemliyoruz
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('purple', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('darkblue', 'red'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('EstimatedSalary')
plt.legend()
plt.show()
# Test sonuçlarını gözlemliyoruz.
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('darkblue', 'purple'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print("ACC: ",accuracy_score(y_pred,y_test))
# Decision Tree Classification
# Veri setini test ve eğitim olarak 2'ye ayırıyoruz.
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#Özellik ölçekleme (Decison Tree için ölçekleme yapmak şart değil ancak görselleştirme kodunu çalıştırırken ölçeklenmiş veriye ihtiyaç duyuluyor.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# eğitim setine Decision Tree algoritmasını uyguluyoruz 
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Test veri setini kullanarak sonuçları tahmin ediyoruz
y_pred = classifier.predict(X_test)
# Confusion matrisimizi oluşturuyoruz.
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
# Eğitim sonuçları gözlemliyoruz
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('purple', 'blue'))(i), label = j)
plt.title('Decision Tree (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
# Test sonuçlarını gözlemliyoruz.
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('Darkblue', 'Skyblue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'yellow'))(i), label = j)
plt.title('Decision Tree (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print("ACC: ",accuracy_score(y_pred,y_test))