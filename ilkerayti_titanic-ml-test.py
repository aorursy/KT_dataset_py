
# ../input/test.csv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# veriler.info()
veriler.head(10)
#verileri oku gereksız oldugu açık olanları sil, nonları keşfet, 
#non ları ya satırı sil veya empty ya da ortala değerle değiştir.
#veriler = pd.read_csv('train.csv')
veriler = veriler.drop(["Name"],axis=1) #asagıdakılerın onemsız oldugunu dusunuyorum
veriler = veriler.drop(["Ticket"],axis=1)
veriler = veriler.drop(["PassengerId"],axis=1)
#veriler = veriler.drop(["Cabin"],axis=1)
veriler = veriler.drop(["Fare"],axis=1)
#veriler.dropna(inplace=True) #sildim
#veriler.info()

#veriler["Cabin"].value_counts(dropna =False) # cok fazla sınıf var dırekt kaldırım
#veriler["Embarked"].value_counts(dropna =False) 

    #Kabin
veriler.Cabin.fillna('0', inplace=True)
veriler.loc[veriler.Cabin.str[0] == 'A', 'Cabin'] = 1
veriler.loc[veriler.Cabin.str[0] == 'B', 'Cabin'] = 2
veriler.loc[veriler.Cabin.str[0] == 'C', 'Cabin'] = 3
veriler.loc[veriler.Cabin.str[0] == 'D', 'Cabin'] = 4
veriler.loc[veriler.Cabin.str[0] == 'E', 'Cabin'] = 5
veriler.loc[veriler.Cabin.str[0] == 'F', 'Cabin'] = 6
veriler.loc[veriler.Cabin.str[0] == 'G', 'Cabin'] = 7
veriler.loc[veriler.Cabin.str[0] == 'T', 'Cabin'] = 8


# Gemiye biniş limanlarını tam sayıya çevirelim
veriler.Embarked.fillna(1, inplace=True)
veriler["Embarked"].value_counts(dropna =False)
#veriler['Embarked'].replace('S', 1, inplace=True)
#veriler['Embarked'].replace('C', 2, inplace=True)
#veriler['Embarked'].replace('Q', 3, inplace=True)
veriler.loc[veriler.Embarked.str[0] == 'S', 'Embarked'] = 1
veriler.loc[veriler.Embarked.str[0] == 'C', 'Embarked'] = 2
veriler.loc[veriler.Embarked.str[0] == 'Q', 'Embarked'] = 3


veriler.head()
print(test["Cabin"].value_counts(dropna =False))
test.head()
test = test.drop(["Name"],axis=1) #asagıdakılerın onemsız oldugunu dusunuyorum
test = test.drop(["Ticket"],axis=1)
#test = test.drop(["PassengerId"],axis=1)
#test = test.drop(["Cabin"],axis=1)
test = test.drop(["Fare"],axis=1)
#veriler.dropna(inplace=True) #sildim
#veriler.info()

#veriler["Cabin"].value_counts(dropna =False) # cok fazla sınıf var dırekt kaldırım
#veriler["Embarked"].value_counts(dropna =False) 

    #Kabin
test.Cabin.fillna('0', inplace=True)
test.loc[test.Cabin.str[0] == 'A', 'Cabin'] = 1
test.loc[test.Cabin.str[0] == 'B', 'Cabin'] = 2
test.loc[test.Cabin.str[0] == 'C', 'Cabin'] = 3
test.loc[test.Cabin.str[0] == 'D', 'Cabin'] = 4
test.loc[test.Cabin.str[0] == 'E', 'Cabin'] = 5
test.loc[test.Cabin.str[0] == 'F', 'Cabin'] = 6
test.loc[test.Cabin.str[0] == 'G', 'Cabin'] = 7
test.loc[test.Cabin.str[0] == 'T', 'Cabin'] = 8


# Gemiye biniş limanlarını tam sayıya çevirelim
test.Embarked.fillna(1, inplace=True)
test["Embarked"].value_counts(dropna =False)
#veriler['Embarked'].replace('S', 1, inplace=True)
#veriler['Embarked'].replace('C', 2, inplace=True)
#veriler['Embarked'].replace('Q', 3, inplace=True)
test.loc[test.Embarked.str[0] == 'S', 'Embarked'] = 1
test.loc[test.Embarked.str[0] == 'C', 'Embarked'] = 2
test.loc[test.Embarked.str[0] == 'Q', 'Embarked'] = 3
test.head()
veriler.describe()
#label ardından one hot encodıng yapıp verılerdekı yerıne yazdım
verilerSex = [0 if 'female' == each else 1 for each in veriler.Sex]
veriler.Sex = verilerSex.copy()

veriler.tail()
#print(data.Species.unique())


testSex = [0 if 'female' == each else 1 for each in test.Sex]
test.Sex = testSex.copy()
test.tail()
# age nan degerlerı ortalama degıstır
#veriler["Age"].value_counts(dropna =False)
#veriler["Age"].fillna(0,inplace=True)
verilerAge = [0 if each >= 60 else 1 if each >= 35 else 2 if each >= 18 else 3 if each >= 12 else 4 if each >= 5 else 5 for each in veriler.Age.copy()]
veriler["Age"] = verilerAge.copy()
#veriler['Age'].fillna(veriler['Age'].median(), inplace=True)

veriler.tail()
testAge = [0 if each >= 60 else 1 if each >= 35 else 2 if each >= 18 else 3 if each >= 12 else 4 if each >= 5 else 5 for each in test.Age.copy()]
test["Age"] = testAge.copy()
#test['Age'].fillna(test['Age'].median(), inplace=True)
test.tail()
veriler.describe()
veriler.corr()
veriler.head(10)
#veriler["Sex"].value_counts(dropna =False)
# sutunları kafaya göre değil geri yayılıma göre sil ama gerı yayılım multılınear gore mı yoksa tumu için mi?
# en ıyısını bul
x = veriler.iloc[:,1:8].values #bağımsız değişkenler
#y = veriler.iloc[:,5:].values #bağımlı değişken
y = veriler.Survived.values
print(x[:5])
print(y[:5])
#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train_before, x_test_before,y_train,y_test = train_test_split(x,y,test_size=0.40, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train_before)
X_test = sc.transform(x_test_before)
# 1. Logistic Regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train) #egitim

y_pred = logr.predict(X_test) #tahmin
#print(y_pred)
#print(y_test)

#karmasiklik matrisi
cm = confusion_matrix(y_test,y_pred)
#print('LR')
# %% cm visualization
import seaborn as sns
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
print("score :",logr.score(X_train, y_train))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('KNN')
#print(cm)
import seaborn as sns
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
print("score :",knn.score(X_train, y_train))

# %%
# find k value
score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(X_train,y_train)
    score_list.append(knn2.score(X_train, y_train))
    
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
# 3. SVC (SVM classifier)
from sklearn.svm import SVC
svc = SVC(kernel='poly')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
#print(cm)
import seaborn as sns
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
print("score :",svc.score(X_train, y_train))
# 4. Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')
#print(cm)
import seaborn as sns
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
print("score :",gnb.score(X_train, y_train))
# 5. Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
#print(cm)
import seaborn as sns
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
print("score :",dtc.score(X_train, y_train))
# 6. Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=500, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)

print('RFC')
#print(cm)
import seaborn as sns
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
print("score :",rfc.score(X_train, y_train))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
print(veriler.head())
print("Train Accuracy :", confusion_matrix(y_train,rfc.predict(X_train)))
print("score :",rfc.score(X_train, y_train))
test.head()
#test.info()
print(X_test)

x_yeni= test.iloc[:,1:8].values #bağımsız değişkenler

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_yeni = sc.fit_transform(x_yeni)

print(x_yeni)
y_pred = rfc.predict(x_yeni)
print(len(y_pred))
print(len(test.PassengerId))
print(y_pred)

import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

warnings.filterwarnings('ignore')

titanic_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')

# veri setinin kopyasini olusturalim.
train = titanic_train.copy()
test = titanic_test.copy()

# egitim setindeki passengerID'yi gerek yok.
del train['PassengerId']

# kategorik degiskenleri etiketleyelim.
label_encoder = LabelEncoder()
train['Sex'] = label_encoder.fit_transform(train['Sex'])
train['Embarked'] = label_encoder.fit_transform(train['Embarked'].fillna('0'))

# Ticket degerlerini duzenleyelim.
# Ticket degerinin ilk karakterini alip etiketliyoruz.
train['Ticket'].fillna('N')
train['Ticket'] = train['Ticket'].apply(lambda x: x[0:1] if isinstance(x, str) else '$')
label_encoder = LabelEncoder()
train['Ticket'] = label_encoder.fit_transform(train['Ticket'])

# kullanmayacagimiz verileri silelim.
del train['Name']
del train['Cabin']

# eksik deger duzenlemesi
train['Age'].fillna(train['Age'].mean(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mean(), inplace=True)

# kategorik degiskenleri etiketleyelim.
test['Sex'] = label_encoder.fit_transform(test['Sex'])
test['Embarked'] = label_encoder.fit_transform(test['Embarked'])

# ayni islemleri test veri seti icin uygulayalim.
test['Ticket'].fillna('N')
test['Ticket'] = test['Ticket'].apply(lambda x: x[0:1] if isinstance(x, str) else '$')
le = LabelEncoder()
test['Ticket'] = le.fit_transform(test['Ticket'])

del test['Name']
del test['Cabin']

test['Age'].fillna(test['Age'].mean(), inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)

y_train = np.reshape(np.asmatrix(train['Survived']), (len(train), -1))
del train['Survived']
X_train = np.reshape(np.asmatrix(train), (len(train), -1))

del test['PassengerId']
X_test = np.reshape(np.asmatrix(test), (len(test), -1))

################################################################################
# kendimiz modellerin dogrulugunu olcebilmek icin
# etiketleri belli olan egitim setini egitim-test diye ikiye ayiralim.
################################################################################


trainX, testX, trainY, testY = train_test_split(X_train, y_train, test_size=0.2)
print()
print ()
print (
    'Sayisal Degerler, Kategorik Ozellikler(sex ve embarked), Ticket ve Eksik Deger Manipulasyonu ile Modellerin Accuracy Orani')
model = LogisticRegression()
model.fit(X_train, y_train)
X_pred_logistic_3 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("LogisticRegression :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = Perceptron()
model.fit(X_train, y_train)
X_pred_perceptron_3 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("Perceptron :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = SGDClassifier()
model.fit(X_train, y_train)
X_pred_sgdclassifier_3 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("SGDClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = GaussianNB()
model.fit(X_train, y_train)
X_pred_gaussian_3 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("GaussianNB :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = KNeighborsClassifier()
model.fit(X_train, y_train)
X_pred_kneighbors_3 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("KNeighborsClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = SVC()
model.fit(X_train, y_train)
X_pred_svc_3 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print("SVC :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = LinearSVC()
model.fit(X_train, y_train)
X_pred_linearsvc_3 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("LinearSVC :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
X_pred_decisiontree_3 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("DecisionTreeClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = GradientBoostingClassifier()
model.fit(X_train, y_train)
X_pred_gradientboosting_3 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("GradientBoostingClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

modelr = RandomForestClassifier()
modelr.fit(X_train, y_train)
X_pred_randomforest_3 = modelr.predict(X_test)
modelr.fit(trainX, trainY)
pred = modelr.predict(testX)
print  ("RandomForestClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

y_pred = model.predict(X_test)

#saving
submission = pd.DataFrame({
    "PassengerId":titanic_test.PassengerId,
    "Survived":y_pred
})
submission.to_csv("submission_2126.csv", index=False)
print(submission.head())
print(submission.info())