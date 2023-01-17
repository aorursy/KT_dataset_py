# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import numpy as np

import pandas as pd 



import seaborn as sns

from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score,roc_curve

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier



from warnings import filterwarnings

filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/diabetes/diabetes.csv")
data.head()
data.info()
data.nunique()   #Unique degerler
data.describe().T
print("satir ve sutun = ", data.shape)

print("Boyut sayisi = ",data.ndim)

print("eleman sayisi = ",data.size)    # neden farkli bilmiyorum? 
data.corr()
# Heatmap correlation cizimi

data.corr()



#correlation map

# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

f,ax = plt.subplots(figsize=(10, 6))

sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="yellow", fmt= '.1f',ax=ax) #ftm noktadan sonra kac hane olacak onu verir

plt.show()
#kac adet 1 ve 0 var



sns.countplot(data.Outcome)

#sns.countplot(kill.manner_of_death)

plt.title("Outcome",color = 'blue',fontsize=15)

plt.show()
data.isnull().sum()   # eksik gozlem var mi? 
import missingno as msno

msno.matrix(data)

plt.show()
x = data.drop(["Outcome"], axis=1)  #"Outcome" disindaki sutunlar bagimsiz degisken_

y = data["Outcome"]                 #"Outcome" ise bagimli degiskendir,   
#x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
x.describe().T
# Test Train 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
from sklearn.linear_model import LogisticRegression

loj_reg = LogisticRegression(penalty='l1').fit(x_train,y_train)

loj_reg



#Not penalty= default olursa skor dusuyor.
#?loj_reg
print("b0 = ",loj_reg.intercept_)   # b0 yani sabit degerimizi aldik

print("Coefs = ",loj_reg.coef_)        # bagimsiz degiskenlerin katsayi degerlerini aldik
y_pred = loj_reg.predict(x_test)     # model uzerinden tahmin etme yaptik
loj_reg.predict_proba(x_train)[0:10]  #ikili cikti uretir, 0 ve 1 oranlarini verir.



# sadece 1 ve sifir olarak donmesini degilde, bunlarin olasilik degerlerini ogrenmek istedik.
log_score = accuracy_score(y_test, y_pred)

print("Logistic_reg_class_SCORE = ",log_score)       

# modeldeki gercek 0-1 ile tahmindeki 0-1 oranlarini karsilastirip, dogru siniflandirma oranimizdir.
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=22).fit(x_train, y_train)

rf_model



#n_estimators=10 defaultur. 22 da daha yuksek skor aldim
y_pred = rf_model.predict(x_test)

rf_score = accuracy_score(y_test, y_pred)

print("random_forest_class_SCORE = ",rf_score)       

# modeldeki gercek 0-1 ile tahmindeki 0-1 oranlarini karsilastirip, dogru siniflandirma oranimizdir.
Importance = pd.DataFrame({"Importance": rf_model.feature_importances_*100},

                         index = x_train.columns)

Importance.sort_values(by = "Importance", 

                       axis = 0, 

                       ascending = True).plot(kind ="barh", color = "g")



plt.xlabel("Değişken Önem Düzeyleri")
knn = KNeighborsClassifier( n_neighbors=4).fit(x_train, y_train)

knn
y_pred = knn.predict(x_test)

knn_score = accuracy_score(y_test, y_pred)

print("KNN_class_SCORE = ", knn_score)       

# modeldeki gercek 0-1 ile tahmindeki 0-1 oranlarini karsilastirip, dogru siniflandirma oranimizdir.
y_probs = knn.predict_proba(x_test)     # tekrardan 0 ve 1'in olasilik degerlerini bu degiskene atadik.

y_probs = y_probs[:,1] 

y_pred = [1 if i > 0.75 else 0 for i in y_probs] 





# if ve for donguleri ile "1" in 0.5'den buyukse 1 diye siniflandirmasini, degilse "0" diye siniflandirmasini istedik. 

# ..ve tekrar y_pred seklinde yeni tahminlerimizi atadik

y_pred = [1 if i > 0.75 else 0 for i in y_probs] 
accuracy_score(y_test, y_pred)    # degistirilen diagnosisden sonraki skorumuz
svm_model = SVC(C=5, degree=7, kernel='linear' ).fit(x_train, y_train)

svm_model



# C=5, degree=7, kernel='linear'yaptim. def degil
y_pred = svm_model.predict(x_test)

svm_score = accuracy_score(y_test, y_pred)

print("SVM_class_SCORE = ", svm_score)       

# modeldeki gercek 0-1 ile tahmindeki 0-1 oranlarini karsilastirip, dogru siniflandirma oranimizdir.
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb_model = nb.fit(x_train, y_train)

nb_model
nb_model.predict(x_test)[:10]
nb_model.predict_proba(x_test)[0:10]  



# ! Tahmini olasilik degerleridir. Ilk sutun "0", ikinci sutun "1" i ifade eder.
y_pred = nb_model.predict(x_test)

nb_score = accuracy_score(y_test, y_pred)

print("NB_class_SCORE = ", nb_score)       

# modeldeki gercek 0-1 ile tahmindeki 0-1 oranlarini karsilastirip, dogru siniflandirma oranimizdir.
cross_val_score(nb_model, x_test, y_test, cv = 20).mean()  



#Dogrulanmis test hatamiz, 20 katmanli cross valide edilmis test hatasi ortalamasidir.
# Model Skorlarinin  Seaborn ile gorsellestirilmesi



indexx = ["Log","RF","KNN","SVM","NB"]

regressions = [log_score,rf_score,knn_score,svm_score,nb_score]



plt.figure(figsize=(8,6))

sns.barplot(x=indexx,y=regressions)

plt.xticks()

plt.title('Model Compare',color = 'orange',fontsize=20)

plt.show()
## conda install -c plotly plotly chart-studio
from plotly.plotly import iplot

import plotly.graph_objs as go

import chart_studio.plotly as py





indexx = ["Log","RF","KNN","SVM","NB"]

regressions = [log_score,rf_score,knn_score,svm_score,nb_score]

# creating trace1

trace1 =go.Scatter(

                    x = indexx,

                    y = regressions,

                    mode = "lines+markers+text",

                    name = "#",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= indexx) #uzerine geldiginde ne gorunsun.



data = [trace1]  # olusturdugumuz veriler listeye atadik



# konumlandirmayi yapar ve isimlendirir.(layout)

layout = dict(title = 'Model Compare',

              xaxis= dict(title= 'Models',ticklen= 15,zeroline= True),

              yaxis= dict(title= 'Scores',ticklen= 15,zeroline= True)

             )

fig = dict(data = data, layout = layout)

iplot(fig)