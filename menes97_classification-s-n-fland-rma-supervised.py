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
# Bize lazım olacak küpüphaneleri alalım

import matplotlib.pyplot as plt # Grafik çizimleri için

import seaborn as sns           # Görselleştirme için
data = pd.read_csv("../input/heart-disease-uci/heart.csv") # datamızı çekiyoruz.

data.head()  # datanın ilk 5 satırını görmemizi sağlar

data.shape
data.info() # data ile ilgili bilgilere erişmek için
data.describe()
#Columnların birbiriyle korelasyonu

plt.figure(figsize=(15,9))

sns.heatmap(data.corr(),cmap='Blues',annot=True) 

plt.show()
# Boxplot

l = data.columns.values

number_of_columns=14

number_of_rows = len(l)-1/number_of_columns

plt.figure(figsize=(number_of_columns,5*number_of_rows))

for i in range(0,len(l)):

    plt.subplot(number_of_rows + 1,number_of_columns,i+1)

    sns.set_style('whitegrid')

    sns.boxplot(data[l[i]],color='green',orient='v')

    plt.tight_layout()

# targetin kaç hastada olduğunu bulabiliriz.

sns.countplot(x="target", data=data)

data.loc[:,'target'].value_counts()

from sklearn.model_selection import train_test_split # Datamızı train ve test olarak bölüyoruz.

x,y = data.loc[:,data.columns != 'target'], data.loc[:,'target']

x_train, x_test, y_train, y_test  =train_test_split(x,y, test_size =0.3 , random_state = 42)



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5) # n_neighbors : K değeridir. Bakılacak eleman sayısıdır.

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)



print('K=5 için doğruluk : ',knn.score(x_test,y_test)) 

# k'yı 1'dan 25'e kadar seçiyoruz ve bizim için en uygun değeri bulalım.



aralık = np.arange(1,25)

train_dogruluk =[]

test_dogruluk = []



for i ,k in enumerate(aralık):

    knn = KNeighborsClassifier(n_neighbors=k)

    # knn ile fit ediyoruz.

    knn.fit(x_train,y_train)

    #train doğruluk

    train_dogruluk.append(knn.score(x_train, y_train))

    # test doğruluk

    test_dogruluk.append(knn.score(x_test, y_test))

    

# Şimdi doğruluk grafiğini çizdireceğiz







plt.figure(figsize=[13,8])

plt.plot(aralık, test_dogruluk, label = 'Test Doğruluğu')

plt.plot(aralık, train_dogruluk, label = 'Training Doğruluğu')

plt.legend()

plt.title('-value VS Doğruluk')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(aralık)

plt.savefig('graph.png')

plt.show()

print("En iyi doğruluk {} with K = {}".format(np.max(test_dogruluk),1+test_dogruluk.index(np.max(test_dogruluk))))



    

    
#SWM

from sklearn.svm import SVC



svc = SVC(random_state = 42)



svc.fit(x_train,y_train)



print("SWM modelinin doğrulu {}".format(svc.score(x_test,y_test)))



result = svc.predict(x_test)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,result)

print(cm)

# NAİVE BAYES

from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()

nb.fit(x_train,y_train)



print("Naive Bayes modelinin doğruluğu {}".format(nb.score(x_test,y_test)))



# Confusion Matrix

result = svc.predict(x_test)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,result)

print(cm)
#Decision Tree



from sklearn.tree import DecisionTreeClassifier



dtc = DecisionTreeClassifier(random_state=42)



dtc.fit(x_train,y_train)



result = dtc.predict(x_test)



from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test,result)

print(cm)



print("Decision Tree modeli doğruluk oranı {}".format(dtc.score(x_test,y_test)))
from sklearn.ensemble import RandomForestClassifier

# RandomForestClassifier sınıfını import ettik



rf = RandomForestClassifier (n_estimators =100 , random_state = 42) 

# n_estimators = Oluşturulacak karar ağacı sayısıdır. Değiştirildiğinde başarı oranıda değişir.

rf.fit(x_train,y_train)



print("Random Forest modeli Doğruluğu {}".format(rf.score(x_test,y_test)))

#Logistic Regression

from sklearn.linear_model import LogisticRegression





#normalizition

x = (x - np.min(x))/(np.max(x)-np.min(x)).values

from sklearn.model_selection import train_test_split # Datamızı train ve test olarak bölüyoruz.

x,y = data.loc[:,data.columns != 'target'], data.loc[:,'target']

x_train, x_test, y_train, y_test  =train_test_split(x,y, test_size =0.3 , random_state = 42)





lr = LogisticRegression(random_state=42)

lr.fit(x_train,y_train)



print("Logistic Regression modeli doğruluğu {}".format(lr.score(x_test,y_test)))


