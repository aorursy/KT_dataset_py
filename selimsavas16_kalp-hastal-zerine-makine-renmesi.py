# Matematik

import numpy as np 
import pandas as pd 

# Görselleştirme

import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("../input/heart-disease-uci/heart.csv")
print("Veri'nin ilk 5 satırı")
data.head()
print("Veri'nin son 5 satırı")
data.tail()
print("Satır/Sütun") 
data.shape
print("Verimiz sütunlarında kaç farklı değer barındırıyor.")
data.nunique()
print("Veri için korelasyon bilgileri") 

# Görselleştirilmiş hai bizim için daha iyi ve anlaşılır olucaktır.
data.corr()
print("Veri hakkında temel bilgiler")
data.describe()
print("Veri hakkında bilgiler")
data.info()
print("Boş değerlerin tablo şeklinde gösterimi")
data.isnull()
print("Boş değerlerin sütun  bazlı toplanması")

data.isnull().sum(axis=0)
print("Sütunların isimlerinin incelenmesi")
data.columns
f, ax = plt.subplots(figsize=(10,10))

# annot = True : renklerin içinde sayılarda yazsın
# linewidths = 5 :aralardaki kırmızı çizginin boyutu
# linecolor = red : aralardaki kırmızı çizginin rengi
# ax = ax : belirlediğim değerleri koyucam

sns.heatmap(data.corr(), annot=True, linewidths=5, linecolor="red", fmt =".1f",ax=ax)
plt.show()
sns.barplot(x="cp",
           y="target",
           data=data,)
sns.catplot(x="cp",
           y="target",
           data=data,
           kind="violin")
sns.catplot(y="age",x="target",data=data,kind="violin")
y= data.target.values

x_data = data.drop(["target"],axis=1)
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
x.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

lr= LogisticRegression()

lr.fit(x_train,y_train)

print("Logistic Regression test başarısı {}".format(lr.score(x_test,y_test)))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3) # k

knn.fit(x_train,y_train)
predict = knn.predict(x_test)
predict
print("KNN test başarısı ({}) test başarısı: {} ".format(3,knn.score(x_test,y_test)))
# find k value
score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.figure(figsize=(20,10))
plt.plot(range(1,15),score_list)

plt.show()
# Belirlediğimiz k değerinden daha başarılı k değerleri varsa modelimizi o k değeri ile eğitebilirdik. Ama biz en başarılı olanı seçmişiz zaten.
from sklearn.svm import SVC

svm = SVC(random_state = 1)
svm.fit(x_train,y_train)
print("SVM nin test başarısı {}".format(svm.score(x_test,y_test)))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train,y_train)
print("Naive Bayes test değerleri {}".format(nb.score(x_test,y_test)))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

print("Decision Tree test değerleri {}".format(dt.score(x_test,y_test)))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier( n_estimators =100, random_state=1)
rf.fit(x_train,y_train)

print("Random Forest test değerleri {}".format(rf.score(x_test,y_test)))
y_pred = rf.predict(x_test)
y_true = y_test
#%% confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)

# %% cm visualization
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

lr_score = lr.score(x_test,y_test)
knn_score = knn.score(x_test,y_test)
svm_score = svm.score(x_test,y_test)
nb_score = nb.score(x_test,y_test)
dt_score = dt.score(x_test,y_test)

models_data =["Lr","Knn","Svm","Nb","Dt"]
score_data = [lr_score,knn_score,svm_score,nb_score,dt_score]

score_data
df = pd.DataFrame({'Models': models_data,'Score':score_data})
df
sns.barplot(x= models_data, y= score_data,palette = sns.cubehelix_palette(len(x)))
# asceding : azalan sıralama
new_index = (df["Score"].sort_values(ascending=False)).index.values

# indexleri değiştir
sorted_data = df.reindex(new_index)
sorted_data
plt.figure(figsize=(10,5)) 

sns.barplot(x="Models", y="Score",data=sorted_data)
