# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib as plt 

import matplotlib.pyplot as plt 

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/german_credit_data.csv", index_col=0)
data.head(10) # Veri kümesi tablosunun ilk 10 öznitelik bilgisini verir.
data.tail() # Veri kümesi tablosunun son 5 bilgisini verir.
data.describe() # Veri kümesine ait string olmayan nümerik değeleri gösterir.
data.info() # Veri kümesi ile ilgili bellek kullanımı ve veri istatistikleri
data.corr() # Veri kümesinde bulunan öznitelikler arasında ki ilişkiyi değerlendirir. Duration ve Credit amount arasında 62% ilişki var
data.shape # Veri kümesinin kaç satır ve sütundan oluştuğu bilgisini verir. Shape bir özelliktir ve metod olmadığı için parantez kullanılmamıştır.
data.columns # Veri kümesine ait sütunlarımızı gösterir.
print(data.nunique()) # Benzersiz türleri bulabiliriz.

data.sample(5) # Veri kümesi içerisinde eksik veri olup olmadığını ilk 5 satır özetine bakarak tespit edebiliriz.

pd.concat([data.isnull().sum(), 100 * data.isnull().sum()/len(data)], 

              axis=1).rename(columns={0:'Missing Records', 1:'Percentage (%)'})  

# Veri kümesinde ki eksik veri sayılarını ve oranlarını gösterir.
data["Saving accounts"].value_counts(dropna = False)

# Saving accounts değişkeni için her farklı değerden kaç tane olduğunu hesaplar.

# dropna = False; Eğer NaN değer var ise göster anlamına gelir.
data["Checking account"].value_counts(dropna = False) 


for column in data.columns:

    data[column].fillna(data[column].mode()[0], inplace=True) 

# Kategorik değişkenlerimizi mode yöntemiyle doldurabiliriz.
data.head(5)
# 2- VERİ GÖRSELLEŞTİRME 



fig = plt.figure(figsize=(7,7))   # Veri kümesinde ki cinsiyet dağılımı

data['Sex'].value_counts().plot(kind = 'pie', autopct='%.1f%%')

plt.xlabel("Cinsiyet (0 = kadın, 1= erkek)")

plt.ylabel(" ", fontsize = 20)

plt.title("Cinsiyete göre Yüzdelik Dağılım")

print("")   
fig = plt.figure(figsize=(7,7))   # Veri kümesinde ki meslek dağılımı

data['Job'].value_counts().plot(kind = 'pie', autopct='%.1f%%')

plt.xlabel("Meslek (0 = vasıfsız ve yerleşik olmayan, 1= vasıfsız ve yerleşik olan, 2=vasıflı, 3=çok yetenekli)")

plt.ylabel(" ", fontsize = 20)

plt.title("Meslek için Yüzdelik Dağılım")

print("") 
n_credits = data.groupby("Purpose")["Age"].count().rename("Count").reset_index()  # Verilen kredilerin hangi amaçla alındığını gösterir.

n_credits.sort_values(by=["Count"], ascending=False, inplace=True)



plt.figure(figsize=(10,6))

bar = sns.barplot(x="Purpose",y="Count",data=n_credits)

bar.set_xticklabels(bar.get_xticklabels(), rotation=70)

plt.ylabel("Number of granted credits")

plt.tight_layout()
n_credits = data.groupby("Age")["Purpose"].count().rename("Count").reset_index()  # Verilen kredilerin miktarını gösterir.

n_credits.sort_values(by=["Count"], ascending=False, inplace=True)



plt.figure(figsize=(10,6))

bar = sns.barplot(x="Age",y="Count",data=n_credits)

bar.set_xticklabels(bar.get_xticklabels(), rotation=70)

plt.tight_layout()
def boxes(x,y,h,r=45):

    fig, ax = plt.subplots(figsize=(15,7))

    box = sns.boxplot(x=x,y=y, hue=h, data=data)

    box.set_xticklabels(box.get_xticklabels(), rotation=r)

    fig.subplots_adjust(bottom=0.2)

    plt.tight_layout()
boxes("Purpose","Credit amount","Sex")
boxes("Housing","Credit amount","Sex",r=0)
data.rename(columns = {'Saving accounts': 'Saving_accounts'}, inplace=True)

data.rename(columns = {'Checking account': 'Checking_account'}, inplace=True)

data.rename(columns = {'Credit amount': 'Credit_amount'}, inplace=True)
data.head(5)
print("Sex' : ",data['Sex'].unique())

print("Housing : ",data['Housing'].unique())

print("Saving_accounts : ",data['Saving_accounts'].unique())

print("Checking_account : ",data['Checking_account'].unique())

print("Purpose : ",data['Purpose'].unique())
# Kategorik Değişkenlerin Dönüştürülmesi



from sklearn import preprocessing   # LabelEncoder için gerekli işlemlerin yapılması

le = preprocessing.LabelEncoder()   # Kategorik sütunları nümerik değerlere dönüştürür

 

data['Sex'] = le.fit_transform(data['Sex'])

data['Housing'] = le.fit_transform(data['Housing'])

data['Saving_accounts'] = le.fit_transform(data['Saving_accounts'])

data['Checking_account'] = le.fit_transform(data['Checking_account'])

data['Purpose'] = le.fit_transform(data['Purpose'])



data.head(5)  # Veri kümemizin nümerik halini görebiliriz.
# 3- KÜTÜPHANELERİN YÜKLENMESİ VE MODELİN UYGULANMASI 



from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

# Modelleri uygulamadan önce veri kümemizi train ve test olarak ayırıyoruz.

# Train: kullanılan veriler, Test: Eğitim için kullanılmayan veriler  # hangi test verisini seçtiğimiz önemli

# Random_state: Her zaman aynı sayıyı üreterek sürekliliği sağlar.



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop('Sex', axis=1), data['Sex'], test_size = 0.25, random_state=45) # Satır; axis=0   Sütun; axis=1



# Veri kümesinin %25'i test, %75'i ise eğitim olarak ayrıldı.

lr = LogisticRegression()

lr.fit(X_train,y_train)

lr_score = lr.score(X_test,y_test)

print("Test Accuracy of LR Algorithm: {:.2f}%".format(lr_score*100))

nb = GaussianNB()

nb.fit(X_train, y_train)

nb_score = nb.score(X_test,y_test)

print("Test Accuracy of Naive Bayes: {:.2f}%".format(nb_score*100))
dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)

dtc_score = dtc.score(X_test, y_test)

print("Decision Tree Test Accuracy {:.2f}%".format(dtc_score*100))

rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)

rf.fit(X_train, y_train)

rf_score = rf.score(X_test, y_test)

print("Random Forest Algorithm Accuracy Score: {:.2f}%".format(rf_score*100))
nn = MLPClassifier()

nn.fit(X_train, y_train)

nn_score = nn.score(X_test, y_test)

print("MLP Classifier Accuracy Score: {:.2f}%".format(nn_score*100))