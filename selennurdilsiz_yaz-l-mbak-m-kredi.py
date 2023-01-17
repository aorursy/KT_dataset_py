import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt 



import warnings

warnings.filterwarnings('ignore')



from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import r2_score

from sklearn.metrics import f1_score



import pandas as pd

data = pd.read_csv("../input/krediVeriseti.csv", delimiter = ";")
data.head(5)
data.describe().T#istatistikler
data.shape #satır ve sütun sayısı
data.info #datadaki kolonlar hakkında boş değer,kolon tipi ve total veriyi döndürür
print(list(data.columns))
data.tail()#son 5 satır
data.sample(6) #rastgele 6 satır
data.corr()
data.evDurumu[data.evDurumu == 'evsahibi'] = 1

data.evDurumu[data.evDurumu == 'kiraci'] = 0



data.telefonDurumu[data.telefonDurumu == 'var'] = 1

data.telefonDurumu[data.telefonDurumu == 'yok'] = 0



data.KrediDurumu[data.KrediDurumu == 'krediver'] = True

data.KrediDurumu[data.KrediDurumu == 'verme'] = False
data= data.astype(float) 
data.corr()
#Korelasyon Gösterim

f,ax = plt.subplots(figsize = (12,9))

sns.heatmap(data.corr(), annot = True, linewidths =.5, fmt = '.2f', ax=ax)

plt.show() 
#Eksik Değer Doldurma

#Null olan öznitelikleri buluyoruz

data.isnull().sum()
#Eksik değer tablosu

def eksik_deger_tablosu(data): 

    mis_val = data.isnull().sum()

    mis_val_percent = 100 * data.isnull().sum()/len(data)

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    mis_val_table_ren_columns = mis_val_table.rename(

    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})

    return mis_val_table_ren_columns
eksik_deger_tablosu(data)
data['KrediDurumu'].value_counts()
# Veri setimizi okuyoruz

X = data.iloc[:, :-1].values

Y = data.iloc[:, -1].values
# Veri setini test ve eğitim olarak 2'ye ayırıyoruz.

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
print ("X_train: ", len(x_train))

print("X_test: ", len(x_test))

print("Y_train: ", len(y_train))

print("Y_test: ", len(y_test))
# Özellik ölçekleme

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
# eğitim setine Naive Bayes uyguluyoruz 

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train, y_train)
# Test veri setini kullanarak sonuçları tahmin ediyoruz

y_pred = nb.predict(x_test)
# Confusion matrisimizi oluşturuyoruz.

from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(y_test, y_pred)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)



nb.fit(X_train, y_train)



y_pred = nb.predict(X_test)



# Summary of the predictions made by the classifier

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# Accuracy score

nb_acc=accuracy_score(y_pred,y_test)

print("ACC: ",accuracy_score(y_pred,y_test))
# Decision Tree Classification

# Veri setini test ve eğitim olarak 2'ye ayırıyoruz.

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
# eğitim setine Decision Tree algoritmasını uyguluyoruz 

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

dt.fit(X_train, y_train)
# Test veri setini kullanarak sonuçları tahmin ediyoruz

y_pred = dt.predict(X_test)
# Confusion matrisimizi oluşturuyoruz.

from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(y_test, y_pred)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)



dt.fit(X_train, y_train)



y_pred = dt.predict(X_test)



# Summary of the predictions made by the classifier

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# Accuracy score

dt_acc=accuracy_score(y_pred,y_test)

print("ACC: ",accuracy_score(y_pred,y_test))
#Random Forest

# Veri setini test ve eğitim olarak 2'ye ayırıyoruz.

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

rf=RandomForestClassifier()

rf.fit(x_train,y_train)

y_pred = rf.predict(x_test)
# Confusion matrisimizi oluşturuyoruz.

from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(y_test, y_pred)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)



rf.fit(X_train, y_train)



y_pred = rf.predict(X_test)



# Summary of the predictions made by the classifier

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# Accuracy score

rf_acc=accuracy_score(y_pred,y_test)

print("ACC: ",accuracy_score(y_pred,y_test))
#KNN

# Veri setini test ve eğitim olarak 2'ye ayırıyoruz.

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=9)

knn.fit(x_train,y_train)

knn_y_pred = knn.predict(x_test)
# Confusion matrisimizi oluşturuyoruz.

from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(y_test, y_pred)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)



knn.fit(X_train, y_train)



y_pred = knn.predict(X_test)



# Summary of the predictions made by the classifier

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# Accuracy score

knn_acc=accuracy_score(y_pred,y_test)

print("ACC: ",accuracy_score(y_pred,y_test))
print('Navive Bayes ACC:',nb_acc)

print('Random Forest ACC:',rf_acc)

print('Decision Tree ACC:',dt_acc)

print('KNN ACC:',knn_acc)
import pickle

pickle.dump(nb, open('Model.pkl','wb'))