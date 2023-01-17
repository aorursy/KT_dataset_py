# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

import statsmodels.formula.api as smf

import seaborn as sns

from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score,roc_curve

import statsmodels.formula.api as smf

from sklearn.linear_model import LogisticRegression

from warnings import filterwarnings

filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")

data = df.copy()

data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)                  # Unnamed: 32 sutunu veriye baktigimizda nan lardan olusuyor ondan drop edelim

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]   # binary yani 0 ile 1 degerlerden olusturmamiz gerekiyor. object lerden olusuyor bunun yerine 0 ile 1 lerden olurmali. cunku bize int veya float lazim

data.head()
data.describe()
y = data.diagnosis.values

x_data = data.drop(["diagnosis"], axis=1)
# x degerlerimiz baktigimizda degerlerin cok buyuk oldugu gorulur. Dolayisiyla verimizi normallestirmemiz gerekiyor



#*** Normalize ***#

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values
X_train, X_test, y_train, y_test = train_test_split(x, y, 

                                                    test_size=0.30, 

                                                    random_state=42)
# statsmodels araciligiyla model kurup fit yapalim. Burda bize modelin anlamliligi ve hangi degiskenin ne kadar etki ettigi bu tablodan cikiyor



loj = sm.Logit(y, x)

loj_model= loj.fit()

loj_model.summary()
from sklearn.linear_model import LogisticRegression

loj = LogisticRegression(solver = "liblinear")

loj_model = loj.fit(x,y)

loj_model
# sabit degeri

loj_model.intercept_
# butun bagimsiz degiskenlerin katsayi degerleri

loj_model.coef_
# tahmini yapalim

y_pred = loj_model.predict(x)
# Gercekte 1 iken 1(PP) olanlar 1 iken 0(PN) olanlar, gercekte 0 iken 1(NP) olanlar 0 iken 0(NN) olanlar

confusion_matrix(y, y_pred)
# accuracy degerine bakalim

accuracy_score(y, y_pred)
# en detayli bir siniflandirma algoritmasinin sonuclarini degerlendirecek ciktilardan biri

print(classification_report(y, y_pred))
# ilk 10 model tahmini

loj_model.predict(x)[0:10]
# yukarda 1 ve 0 verdigi degerlerden ziyade asil degerlerini versin istiyorsak 'predict_proba' modulunu kullanarak gercek degerleri

# matriste 0. indexinde veya sol tarafi 0 a ait degerleri, 1. indexinde veya sag tarafi 1 e ait degerleri verir 

loj_model.predict_proba(x)[0:10][:,0:2]                # ilk 10
# simdi yukardaki 'predict_proba' on tahmin olasilik degerlerini model haline getirmeye calisalim

y_probs = loj_model.predict_proba(x)

y_probs = y_probs[:,1]
y_probs[0:10]               # ilk 10
# burdaki tahmin degerlerimizi donguye sokup 0.5 ten buyuklere 1 ve kucuk olanlara 0 versin

y_pred = [1 if i > 0.5 else 0 for i in y_probs]
# yukardaki degere baktigimizda degisikligi farketmis oluruz ama burda degisiklik yok cunku dogrulanmasi gereken cok bir deger yokmus demekki. Bunu yapma amacimiz modelimizi dogrulamaktir.

y_pred[0:10]
confusion_matrix(y, y_pred)
accuracy_score(y, y_pred)
print(classification_report(y, y_pred))
# bunu yukarda yaptik ilk 5 eleman gorunsun

loj_model.predict_proba(x)[:,1][0:5]
logit_roc_auc = roc_auc_score(y, loj_model.predict(x))
fpr, tpr, thresholds = roc_curve(y, loj_model.predict_proba(x)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Oranı')

plt.ylabel('True Positive Oranı')

plt.title('ROC')

plt.show()

# mavi cizgi kurmus oldugumuz model ile ilgili basarimizin grafigi

# kirmizi cizgi hicbirsey yapmasak modelimiz bu sekilde olacak





# Sekilde goruldugu gibi cok degistirilmesi veya dogrulanmasi gereken deger bulamadi bu veride.



# test train ayirma islemine tabi tutalim

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)
# Modelimizi olusturup fit edelim

loj = LogisticRegression(solver = "liblinear")

loj_model = loj.fit(X_train,y_train)

loj_model
# dogrulanma skorunu bulalim

accuracy_score(y_test, loj_model.predict(X_test))
# dogrulanmis modelin CV skoru bulalim

cross_val_score(loj_model, X_test, y_test, cv = 10).mean()
# model kurma

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn_model = knn.fit(X_train, y_train)

knn_model
# tahmin degeri

y_pred = knn_model.predict(X_test)
accuracy_score(y_test, y_pred)
# detayli ciktimizida alalim. 

print(classification_report(y_test, y_pred))
# KNN parametrelerini bulma

knn_params = {"n_neighbors": np.arange(1,50)}
# siniflandirmasi ve CV ile fit yapalim

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, knn_params, cv=10)

knn_cv.fit(X_train, y_train)
# bunu sadece gozlemlemek icin yapiyoruz. Final modeli onemli bizim icin

print("En iyi skor:" + str(knn_cv.best_score_))

print("En iyi parametreler: " + str(knn_cv.best_params_))
# yukarida ciktida ortaya cikan n_neighbors 11 cikmisti bunu kullanarak KNN olusturulup tuned edelim

knn = KNeighborsClassifier(11)

knn_tuned = knn.fit(X_train, y_train)
# simdide test in tuned score una bakalim

knn_tuned.score(X_test, y_test)
# tahmin degeri

y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
# model ve nesne olusturma fit ile beraber yapalim

from sklearn.svm import SVC



svm_model = SVC(kernel = "linear").fit(X_train, y_train)
svm_model
y_pred = svm_model.predict(X_test)
accuracy_score(y_test, y_pred)
# C parametresi olusturulacak olan dogrunun veya ayrimin olusturulmasiyla ilgili bir kontrol etme imkani saglayan parametredir

# C degeri 0 olamaz hata verir ondan 1 den baslasin



svc_params = {"C": np.arange(1,10)}
svc = SVC(kernel = "linear")


svc_cv_model = GridSearchCV(svc,svc_params, 

                            cv = 10, 

                            n_jobs = -1, 

                            verbose = 2 )



svc_cv_model.fit(X_train, y_train)
# en iyi parametre degerleri

print("En iyi parametreler: " + str(svc_cv_model.best_params_))
# tuned edip fit leyelim

svc_tuned = SVC(kernel = "linear", C = 5).fit(X_train, y_train)
# simdi gercek deger ile tahmin edilen degerin karsilastirma islemini yapalim

y_pred = svc_tuned.predict(X_test)

accuracy_score(y_test, y_pred)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

nb_model = nb.fit(X_train, y_train)

nb_model
# tahmin islemini yapalim

nb_model.predict(X_test)[0:10]
y_pred = nb_model.predict(X_test)
accuracy_score(y_test, y_pred)
cross_val_score(nb_model, X_test, y_test, cv = 10).mean()