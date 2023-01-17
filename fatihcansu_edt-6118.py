# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#datayı okuyalım.

data=pd.read_excel("../input/datum.xlsx")
# datanın ilk 5 satırı..

data.head()
# datanın kolon isimleri ve her bir kolondaki değisken tipleri ve kaçar tane olduğunu aşagıdaki kod ile gorebiliriz.

data.info()
#kategori kolonundaki her bir değer NaN olduğu için bu kolonu siliyoruz.

data.drop(["KATEGORİ"],axis=1,inplace=True)
# Kurs kolonlarındaki ALDI/ALMADI yerine 1/0 yazalım.

data["Fen kurs"]=[1 if i=="ALDI" else 0 for i in data["Fen kurs"]]

data["Matematik kurs"]=[1 if i=="ALDI" else 0 for i in data["Matematik kurs"]]

data["Türkce kurs"]=[1 if i=="ALDI" else 0 for i in data["Türkce kurs"]]

data["Yabancı dil kurs"]=[1 if i=="ALDI" else 0 for i in data["Yabancı dil kurs"]]

data["Drama kurs"]=[1 if i=="ALDI" else 0 for i in data["Drama kurs"]]

data["Beden egitimi kurs"]=[1 if i=="ALDI" else 0 for i in data["Beden egitimi kurs"]]

data["Müzik kurs"]=[1 if i=="ALDI" else 0 for i in data["Müzik kurs"]]

data["Görsel sanatlar kurs"]=[1 if i=="ALDI" else 0 for i in data["Görsel sanatlar kurs"]]

data["Yazarlık ve yazma becerileri kurs"]=[1 if i=="ALDI" else 0 for i in data["Yazarlık ve yazma becerileri kurs"]]
# "oda" ve "sürekli hastalık" kolonları için  Yok/Var yerine 0/1 yazalım.

data["Oda"]=[1 if i=="VAR" else 0 for i in data["Oda"]]

data["Sürekli hastalık"]=[1 if i=="VAR" else 0 for i in data["Sürekli hastalık"]]
# datanın son haline bir bakalım.

data.head()
data.GBO.value_counts() # GBO kolonunda hangi degerden kac tane var bunu verir.
data['GBO'].describe()
data.corr()['GBO'].sort_values()
data.corr().abs()['GBO'].sort_values(ascending=False)
# Aykırı değer tespiti

sns.boxplot(x=data['GBO'])

plt.show()
# GBO degerlerinin histogramı

data.GBO.plot(kind = 'hist',figsize = (8,8))

plt.xlabel("GBO")

plt.grid()

plt.title('GBO  Histogramı',color = 'red',fontsize=15)

plt.show()
# GBO degerlerinin bar grafigi

plt.figure(figsize=(15,10))

plt.bar(data['GBO'].value_counts().index, 

        data['GBO'].value_counts().values,

         fill = 'navy', edgecolor = 'k', width = 1)

plt.xlabel('GBO'); plt.ylabel('Count'); plt.title('GBO Dagılımı');

plt.xticks(list(range(31, 100)))

plt.show()

plt.figure(figsize=(15,10))

sns.countplot(data.GBO)

plt.grid()

plt.title("GBO",color="red",fontsize=15)
# GBO degerlerinin histogramı

data.GBO.plot(kind = 'hist', bins=4, figsize = (8,8))

plt.xlabel("GBO")

plt.grid()

plt.title('GBO  Histogramı',color = 'red',fontsize=15)

plt.show()
data["Cins"]=[1 if i=="E" else 0 for i in data["Cins"]]

data["Anne sag/ölü"]=[1 if i=="SAĞ" else 0 for i in data["Anne sag/ölü"]]

data["Baba sag/ölü"]=[1 if i=="SAĞ" else 0 for i in data["Baba sag/ölü"]]

data["Anne baba birlikte/ayrı"]=[1 if i=="BİRLİKTE" else 0 for i in data["Anne baba birlikte/ayrı"]]

data["Anne calısma durumu"]=[1 if i=="EVET" else 0 for i in data["Anne calısma durumu"]]

data["Baba calısma durumu"]=[1 if i=="EVET" else 0 for i in data["Baba calısma durumu"]]
data.head()

data["Aile ile yasama durumu"].value_counts().unique
data["Anne ogrenim durumu"].value_counts().unique
data["Baba ogrenim durumu"].value_counts().unique
d = {'ANNESİYLE': 0, 'BABASIYLA': 1, 'BABAANNESİYLE': 2, "AİLESİYLE": 3}

data['Aile ile yasama durumu'] = data['Aile ile yasama durumu'].map(d)



d = {'YOK': 0, "İLKOKUL": 1, "ORTAOKUL": 2 , 'ORTA': 2, 'LİSE': 3, "ÖNLİSANS" : 4, "LİSANS": 5, "LİSANSÜSTÜ": 6}

data['Anne ogrenim durumu'] = data['Anne ogrenim durumu'].map(d)

data['Baba ogrenim durumu'] = data['Baba ogrenim durumu'].map(d)



data.info()
data.head()
data.nunique()
f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(data.corr(), annot=True, linewidths=.4, fmt= '.1f',ax=ax)

plt.show()
data_new=data.copy()

data_new=data_new.loc[:, ["Yas","Cins","Devam","Anne ogrenim durumu", "Baba ogrenim durumu", "Kardes","Oda","GBO"]]

data_new.head()
def sırala(item):

    if 31<=item<= 48:

        return 1

    elif 49<=item <= 67:

        return 2

    elif 68<=item <= 82:

        return 3

    else:

        return 4



data_new['GBO'] = data_new['GBO'].apply(sırala)
data_new.head()
f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(data_new.corr(), annot=True, linewidths=.4, fmt= '.1f',ax=ax)

plt.show()
y = data_new.GBO.values

x_data = data_new.drop(["GBO"],axis=1)
from sklearn.model_selection import train_test_split

from sklearn import tree

X_train, X_test, y_train, y_test = train_test_split(x_data, y, test_size=0.3, random_state=100)
grade_classifier = tree.DecisionTreeClassifier(max_leaf_nodes=len(x_data.columns), random_state=0)

grade_classifier.fit(X_train, y_train)
predictions = grade_classifier.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_true = y_test, y_pred = predictions)
from sklearn.ensemble import RandomForestClassifier



rf=RandomForestClassifier(n_estimators=100,random_state=11)



rf.fit(X_train,y_train)



print("random forest algo score: ",rf.score(X_test,y_test))
prediction_1=rf.predict(X_test)

prediction_1
data1=pd.DataFrame({"real_values": y_test, "prediction_values": prediction_1})

data1.head()
#import train_test_split to find train and test sets

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(x_data, y, test_size=0.25, random_state = 1)
print("X_train shape: "+str(X_train.shape))

print("y_train shape: "+str(y_train.shape))

print("X_test shape: "+str(X_test.shape))

print("y_test shape: "+str(y_test.shape))
import itertools



from sklearn.model_selection import StratifiedKFold



from sklearn.svm import LinearSVC



from sklearn import metrics

from sklearn.metrics import precision_recall_fscore_support as score
svm_model = LinearSVC(penalty='l2', random_state=0, tol=0.00001,max_iter=4500)

svm_model.fit(X_train,y_train)
y_predict2 = svm_model.predict(X_test)



print("Accuracy:\n", metrics.accuracy_score(y_test,y_predict2))

print("Confusion Matrix:\n",metrics.confusion_matrix(y_test, y_predict2))
print(y_predict2)
unique, counts = np.unique(y_test, return_counts=True)



unique, counts
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion Matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()

    

# Compute confusion matrix

cnf_matrix = metrics.confusion_matrix(y_test, y_predict2)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure(figsize=(12,6))

plot_confusion_matrix(cnf_matrix, classes=['1','2', '3', '4'])