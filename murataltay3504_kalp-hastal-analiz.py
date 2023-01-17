import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns;sns.set(style="ticks", color_codes=True)

from datetime import datetime



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Imputer

from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc

import os

import warnings

warnings.filterwarnings('ignore')

print(os.listdir("../input"))

data=pd.read_csv('../input/heart.csv')
print("ilk 5 değer analizi için \n")

data.head()
print('Son 5 veri\n')

data.tail()
print('Veri Hakkında Detay \n')

data.describe()
print('Veri Hakkında Bilgi \n')

data.info()


print('Verinin Kolonlarını İnceliyoruz :\n')

data.columns
data.sample(frac=0.05)
data.sample(5)
data=data.rename(columns={'age':'Age','sex':'Sex','cp':'Cp','trestbps':'Trestbps','chol':'Chol','fbs':'Fbs','restecg':'Restecg','thalach':'Thalach','exang':'Exang','oldpeak':'Oldpeak','slope':'Slope','ca':'Ca','thal':'Thal','target':'Target'})
data.columns
data.shape
# Şimdi, tüm verileri null olarak kontrol edeceğim ve eğer veriler null olursa null verinin toplamını alacağım. Bu şekilde, verilerde kaç tane eksik veri bulunduğunu gösterir.

print('Verideki Null değerleri topla \n')



data.isnull().sum()
#tüm satırlar null değerleri kontrol eder

data.isnull().values.any()
plt.figure(figsize=(10,10))

sns.heatmap(data.corr(),annot=True, fmt=".1f")

plt.show()
plt.figure(figsize=(10,10))

sns.heatmap(data.corr(), vmax=.3, center=0,

           square=False, linewidth=.5, cbar_kws={"shrink": .5}, annot=True)

plt.tight_layout()

plt.show()
g =sns.pairplot(data)

plt.show()
data.Age.value_counts()[:10]
data.head(10)
sns.barplot(x=data.Age.value_counts()[:10].index, y=data.Age.value_counts()[:10].values)

plt.xlabel("Age")

plt.ylabel("Age Counter")

plt.title("Age Analysis System")

plt.show()
#min ve max yaşları bulalım

minAge=min(data.Age)

maxAge=max(data.Age)

meanAge= data.Age.mean()

print("Minumum yaş :",minAge)

print("Maksimum yaş : ",maxAge)

print("Yaş Ortalaması :",meanAge)

young_ages=data[(data.Age>=29)&(data.Age<40)]

middle_ages=data[(data.Age>=40)&(data.Age<55)]

elderly_ages=data[(data.Age>55)]

print('Young Ages :',len(young_ages))

print('Middle Ages :',len(middle_ages))

print('Elderly Ages :',len(elderly_ages))
sns.barplot(x=["young ages","middle ages","elderly ages"], y=[len(young_ages),len(middle_ages),len(elderly_ages)])

plt.xlabel("Age Range")

plt.ylabel("Age Counts")

plt.title("Ages State in Dataset")

plt.show()
data.head()
data["AgeRange"]=0

youngAge_index = data [(data.Age>=29)& (data.Age<40)].index

middleAge_index = data[(data.Age>=40)&(data.Age<55)].index

elderlyAge_index = data[(data.Age>55)].index
for index in elderlyAge_index:

    data.loc[index,"AgeRange"]=2

for index in middleAge_index:

    data.loc[index,"AgeRange"]=1

for index in youngAge_index:

    data.loc[index,"AgeRange"]=0
data.head()
sns.swarmplot(x="AgeRange", y="Age", hue="Sex",

             palette=["r","c","y"],

             data=data)

plt.show()
sns.violinplot(data.Age, palette="Set3", bw=.2, cut=1, linewidth=4)

plt.xticks(rotation=45)

plt.title("Age Rates")

plt.show()
data.Sex.value_counts(),
#Cinsiyet (1 = Erkek; 0 = Kadın)

sns.countplot(data.Sex)

plt.show()
total_genders_count=len(data.Sex)

male_count = len(data[data["Sex"]==1])

female_count=len(data[data["Sex"]==0])

print("Toplam Sayı  :",total_genders_count)

print("Erkek Sayısı :",male_count)

print("Kadın Sayısı :",female_count)
#Yüzde oranları

print("Erkek Durumu : {:.2f}%".format((male_count / (total_genders_count)*100)))

print("Kadın Durumu : {:.2f}%".format((female_count / (total_genders_count)*100)))
data.head()
male_andtarget_on=len(data[(data.Sex==1)&(data.Target==1)])

male_andtarget_off=len(data[(data.Sex==1)&(data.Target==0)])

sns.barplot(x=["Male Target On","Male Target Off"], y=[male_andtarget_on,male_andtarget_off])

plt.xlabel("Erkek ve Hedef Durum")

plt.ylabel("Sayısı")

plt.title("Cinsiyet Durumu")

plt.show()
female_andtarget_on=len(data[(data.Sex==0)&(data.Target==1)])

female_andtarget_off=len(data[(data.Sex==0)&(data.Target==0)])

sns.barplot(x=["Female Target On","Female Target Off"], y=[female_andtarget_on, female_andtarget_off])

plt.xlabel("Kadın ve Hedef Durum")

plt.ylabel("Sayısı")

plt.title("Cinsiyet Durumu")

plt.show()
sns.relplot(x="Trestbps", y="Age",

          sizes=(40,400), alpha=.5, palette="muted",

           height=6, data=data

          )

plt.show()
data.head()
# Görüldüğü gibi, 4 tip göğüs ağrısı vardır.

data.Cp.value_counts()
sns.countplot(data.Cp)

plt.xlabel("Ağrı Tipi")

plt.ylabel("Sayısı")

plt.title("Ağrı tipi ve durum sayısı")

plt.show()

# 0 durum en az

# 1 durum biraz sıkıntılı

# 2 koşul orta sorun

# 3 durum çok kötü
cp_zero_target_zero= len(data[(data.Cp==0)&(data.Target==0)])

cp_zero_target_one= len(data[(data.Cp==0)&(data.Target==1)])
sns.barplot(x=["CP 0 Target O","CP 0 Tarfget 1"], y=[cp_zero_target_zero,cp_zero_target_one])

plt.show()
cp_one_target_zero=len(data[(data.Cp==1)&(data.Target==0)])

cp_one_target_one=len(data[(data.Cp==1)&(data.Target==1)])
sns.barplot(x=["CP 1 Target 0","Cp 1 Target 1"], y=[cp_one_target_zero,cp_one_target_one])

plt.show()
cp_two_target_zero=len(data[(data.Cp==2)&(data.Target==0)])

cp_two_target_one=len(data[(data.Cp==2)&(data.Target==1)])
sns.barplot(x=['cp_two_target_zero','cp_two_target_one'],y=[cp_two_target_zero,cp_two_target_one])

plt.show()
cp_three_target_zero=len(data[(data.Cp==3)&(data.Target==0)])

cp_three_target_one=len(data[(data.Cp==3)&(data.Target==1)])
sns.barplot(x=['cp_three_target_zero','cp_three_target_one'],y=[cp_three_target_zero,cp_three_target_one])

plt.show()
target_0_agerang_0=len(data[(data.Target==0)&(data.AgeRange==0)])

target_1_agerang_0=len(data[(data.Target==1)&(data.AgeRange==0)])
colors= ["blue","yellow"]

explode= [0,0]

plt.figure(figsize=(7,7))

plt.pie([target_0_agerang_0,target_1_agerang_0], explode=explode, labels=["Target 0 Age Range 0","Target 1 Age Range 0"], colors=colors, autopct="%1.1f%%")

plt.title("Gençlere Göre Target vs Age Range", color="blue", fontsize=15)

plt.show()
target_0_agerang_1=len(data[(data.Target==0)&(data.AgeRange==1)])

target_1_agerang_1=len(data[(data.Target==1)&(data.AgeRange==1)])
colors= ["red","yellow"]

explode= [0,0]

plt.figure(figsize=(7,7))

plt.pie([target_0_agerang_1,target_1_agerang_1], explode=explode, labels=["Target 0 Age Range 1","Target 1 Age Range 1"], colors=colors, autopct="%1.1f%%")

plt.title("Orta Yaşlılara Göre Target vs Age Range")

plt.show()
colors = ['red','yellow']

explode = [0,0]

plt.figure(figsize = (5,5))

plt.pie([target_0_agerang_1,target_1_agerang_1], explode=explode, labels=['Target 0 Age Range 1','Target 1 Age Range 1'], colors=colors, autopct='%1.1f%%')

plt.title('Target vs Age Range Middle Age',color = 'blue',fontsize = 15)

plt.show()
target_0_agerang_2=len(data[(data.Target==0)&(data.AgeRange==2)])

target_1_agerang_2=len(data[(data.Target==1)&(data.AgeRange==2)])
colors= ["red","blue"]

explode= [0,0]

plt.figure(figsize= (5,5))

plt.pie([target_0_agerang_2, target_1_agerang_2], explode=explode, labels=["Target 0 Age Range 2","Target 1 Age Range 2"], colors=colors, autopct= "%1.1f%%")

plt.title("Yaşlılara Göre Target vs Age Range")

plt.show()
colors = ['red','blue']

explode = [0,0]

plt.figure(figsize = (5,5))

plt.pie([target_0_agerang_2,target_1_agerang_2], explode=explode, labels=['Target 0 Age Range 2','Target 1 Age Range 2'], colors=colors, autopct='%1.1f%%')

plt.title('Target vs Age Range Elderly Age ',color = 'blue',fontsize = 15)

plt.show()
data.head()
data.Thalach.value_counts()[:20]

#ilk 20 satır
sns.barplot(x=data.Thalach.value_counts()[:20].index, y=data.Thalach.value_counts()[:20].values)

plt.xlabel("Thalach")

plt.ylabel("Sayısı")

plt.title("Thalach Sayıları")

plt.xticks(rotation=45)

plt.show()
mean_thalach
age_unique= sorted(data.Age.unique())

age_thalach_values= data.groupby("Age")["Thalach"].count().values

mean_thalach=[]

for i,age in enumerate(age_unique):

    mean_thalach.append(sum(data[data["Age"]==age].Thalach)/age_thalach_values[i])
plt.figure(figsize=(10,5))

sns.pointplot(x=age_unique, y=mean_thalach, color="red", alpha=0.8)

plt.xlabel("Age", fontsize=15, color="blue")

plt.xticks(rotation=45)

plt.ylabel("Thalach", fontsize=15, color="blue")

plt.title("Age vs Thalach", fontsize=15, color="blue")

plt.grid()

plt.show()


age_range_thalach= data.groupby("AgeRange")["Thalach"].mean()
sns.barplot(x=age_range_thalach.index, y=age_range_thalach.values)

plt.xlabel("Age Range Values")

plt.ylabel("Maximum Thalch By Age Range")

plt.title("İllustration of the thalach to the")
sns.barplot(x=age_range_thalach.index,y=age_range_thalach.values)

plt.xlabel('Age Range Values')

plt.ylabel('Maximum Thalach By Age Range')

plt.title('thalachın yaş aralığında gösterimi')

plt.show()
cp_thalach=data.groupby('Cp')['Thalach'].mean()
sns.barplot(x=cp_thalach.index,y=cp_thalach.values)

plt.xlabel("Göğüs Ağrısı Derecesi(Cp)")

plt.ylabel('Cp değerlerine göre maksimum thalach')

plt.title("Göğüs ağrısı derecesine göre thalach gösterimi")

plt.show()

#Bu grafikte görüldüğü gibi, kalp atış hızının daha az olduğu görülüyor

# göğüs ağrısı azaldığında Ancak göğüs ağrısının olduğu durumlarda

# 1, alanın daha fazla olduğu gözlenmiştir. 2 ve 3 bulundu

# aynı derecede.
data.Thal.value_counts()
sns.countplot(data.Thal)

plt.show()
data[(data.Thal==0)]

#Görüldüğü gibi, sadece% 50'sinin% 50 hedef olduğu anlaşıldı.
data[(data["Thal"]==1)].Target.value_counts()

sns.barplot(x=data[(data["Thal"]==1)].Target.value_counts().index, y= data[(data["Thal"]==1)].Target.value_counts().values)

plt.xlabel("Thal Değeri")

plt.ylabel("Sayısı")

plt.title("Thal Sayacı")

plt.show()
#Target 1

a=len(data[(data['Target']==1)&(data['Thal']==0)])

b=len(data[(data['Target']==1)&(data['Thal']==1)])

c=len(data[(data['Target']==1)&(data['Thal']==2)])

d=len(data[(data['Target']==1)&(data['Thal']==3)])

print('Target 1 Thal 0: ',a)

print('Target 1 Thal 1: ',b)

print('Target 1 Thal 2: ',c)

print('Target 1 Thal 3: ',d)



print('*'*50)



e=len(data[(data['Target']==0)&(data['Thal']==0)])

f=len(data[(data['Target']==0)&(data['Thal']==1)])

g=len(data[(data['Target']==0)&(data['Thal']==2)])

h=len(data[(data['Target']==0)&(data['Thal']==3)])

print('Target 0 Thal 0: ',e)

print('Target 0 Thal 1: ',f)

print('Target 0 Thal 2: ',g)

print('Target 0 Thal 3: ',h)
f,ax=plt.subplots(figsize=(7,7))

sns.barplot(y=['T 1&0 Th 0','T 1&0 Th 1','T 1&0 Th 2','Ta 1&0 Th 3'],x=[1,6,130,28],color='green',alpha=0.5,label='Target 1 Thal Durumu')

sns.barplot(y=['T 1&0 Th 0','T 1&0 Th 1','T 1&0 Th 2','Ta 1&0 Th 3'],x=[1,12,36,89],color='red',alpha=0.7,label='Target 0 Thal Durumu')

ax.legend(loc='lower right',frameon=True)

ax.set(xlabel='Target State and Thal Sayacı',ylabel='Target State and Thal Durumu',title='Target VS Thal')

plt.xticks(rotation=90)

plt.show()

#bu yüzden çok güzel bir grafik ekran oldu. Durumu en iyi tanımlayan durum budur.
data.Target.unique()
sns.countplot(data.Target)

plt.xlabel('Target')

plt.ylabel('Sayısı')

plt.title('Target Sayacı 1 & 0')

plt.show()
#Hastalığı olan ve olmayan hastaların yaş aralıklarını belirler ve bunlar hakkında analizler yapar

age_counter_target_1=[]

age_counter_target_0=[]

for age in data.Age.unique():

    age_counter_target_1.append(len(data[(data['Age']==age)&(data.Target==1)]))

    age_counter_target_0.append(len(data[(data['Age']==age)&(data.Target==0)]))

   


plt.scatter(x=data.Age.unique(),y=age_counter_target_1,color='blue',label='Target 1')

plt.scatter(x=data.Age.unique(),y=age_counter_target_0,color='red',label='Target 0')

plt.legend(loc='upper right',frameon=True)

plt.xlabel('Age')

plt.ylabel('Count')

plt.title('Target 0 & Target 1 State')

plt.show()
sns.lineplot(x="Sex", y="Oldpeak",

             hue="Target",data=data)

plt.show()
data.head()
data.head()
g = sns.catplot(x="AgeRange", y="Chol",

                 hue="Sex",

                 data=data, kind="bar",

                 height=4, aspect=.7)

plt.show()
ax = sns.barplot("Sex", "Chol", data=data,

                  linewidth=2.5, facecolor=(1, 1, 1, 0),

                  errcolor=".2", edgecolor=".2")

plt.show()
male_young_t_1=data[(data['Sex']==1)&(data['AgeRange']==0)&(data['Target']==1)]

male_middle_t_1=data[(data['Sex']==1)&(data['AgeRange']==1)&(data['Target']==1)]

male_elderly_t_1=data[(data['Sex']==1)&(data['AgeRange']==2)&(data['Target']==1)]

print(len(male_young_t_1))

print(len(male_middle_t_1))

print(len(male_elderly_t_1))
f,ax1=plt.subplots(figsize=(20,10))

sns.pointplot(x=np.arange(len(male_young_t_1)),y=male_young_t_1.Trestbps,color='lime',alpha=0.8,label='Young')

sns.pointplot(x=np.arange(len(male_middle_t_1)),y=male_middle_t_1.Trestbps,color='black',alpha=0.8,label='Middle')

sns.pointplot(x=np.arange(len(male_elderly_t_1)),y=male_elderly_t_1.Trestbps,color='red',alpha=0.8,label='Elderly')

plt.xlabel('Range',fontsize = 15,color='blue')

plt.xticks(rotation=90)

plt.legend(loc='upper right',frameon=True)

plt.ylabel('Trestbps',fontsize = 15,color='blue')

plt.title('Age Range Values vs Trestbps',fontsize = 20,color='blue')

plt.grid()

plt.show()