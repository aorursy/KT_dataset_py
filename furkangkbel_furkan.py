# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
      

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_india = pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv")
df_individual = pd.read_csv("/kaggle/input/covid19-in-india/IndividualDetails.csv")
df_population = pd.read_csv("/kaggle/input/covid19-in-india/population_india_census2011.csv")
df_imcrt = pd.read_csv("/kaggle/input/covid19-in-india/ICMRTestingLabs.csv")
df_hospital_bed = pd.read_csv("/kaggle/input/covid19-in-india/HospitalBedsIndia.csv")
df_age_group = pd.read_csv("/kaggle/input/covid19-in-india/AgeGroupDetails.csv")


df_india['ConfirmedIndianNational'] = df_india.ConfirmedIndianNational.replace('-',0)
df_india['ConfirmedForeignNational'] = df_india.ConfirmedForeignNational.replace('-',0)
df_india['ConfirmedIndianNational']= df_india['ConfirmedIndianNational'].astype('int64')
df_india['ConfirmedForeignNational']= df_india['ConfirmedForeignNational'].astype('int64')
df_india['Total Cases'] = df_india['Confirmed']
df_india['Active Cases'] = df_india['Total Cases'] - df_india['Cured'] - df_india['Deaths']
df_india["Date"] = pd.to_datetime(df_india["Date"],infer_datetime_format=True,dayfirst=True)
recent_date = df_india['Date'].max()
covid_19_india = df_india[df_india['Date']==recent_date]
covid_19_india[["Date",'State/UnionTerritory','Confirmed','Cured','Deaths']]
df_india['State/UnionTerritory'].nunique()
df_india.shape
df_india=df_india.drop(["Sno"],axis=1)
df_india.isnull().sum()
df_india_date_wise = df_india.groupby(['Date','State/UnionTerritory','Total Cases'])['Cured','Deaths','Active Cases','Confirmed'] \
            .sum().reset_index().sort_values('Total Cases',ascending = False)
recent_date = df_india['Date'].max()
print(recent_date)
df_india_date_wise
plt.figure(figsize=(20,10))

plt.subplot(2,1,1)
plt.bar(df_india_date_wise["State/UnionTerritory"],df_india_date_wise["Deaths"],color='chocolate')
plt.xticks(rotation=75)
plt.ylim(0,3000)
plt.xlabel("Eyaletler")
plt.ylabel("Ölümle Sonuçlanan Vakitler")

plt.subplot(2,1,2)
plt.bar(df_india_date_wise["State/UnionTerritory"],df_india_date_wise["Cured"],color='darkolivegreen')
plt.xlabel("Eyaletler")
plt.ylabel("İyileşen Vakalar")
plt.ylim(0, 35000)

plt.xticks(rotation=75)

plt.show()
#aktif vakalar
plt.figure(figsize=(30,20))
plt.bar(df_india_date_wise["State/UnionTerritory"],df_india_date_wise["Active Cases"])
plt.xticks(rotation=75)
plt.show()
tmp = df_india_date_wise.groupby(['Date'])["Active Cases","Cured","Deaths","State/UnionTerritory","Confirmed"].sum().reset_index()

tmp.tail(1000)
plt.figure(figsize=(20,10))
sns.barplot(tmp["Date"],tmp["Active Cases"],color="b")
plt.xticks(rotation=75)
plt.xlabel("Tarih")
plt.ylabel("Aktif Vakalar")
plt.show()
plt.figure(figsize=(20,10))
sns.lineplot(tmp["Date"],tmp["Deaths"],estimator="median")
sns.lineplot(tmp["Date"],tmp["Active Cases"],estimator="median")
sns.lineplot(tmp["Date"],tmp["Cured"],estimator="median")
plt.xticks(rotation=75)
plt.legend(["Ölümler","Onaylanan Vakalar","İyileşen Vakalar"])
plt.show()
plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
sns.barplot(df_india["Date"],df_india["Deaths"],color="b")
plt.xlabel("Tarih")
plt.ylabel("Ölümlü Vakalar")
plt.xticks(rotation=75)
plt.subplot(2,1,2)
sns.barplot(df_india["Date"],df_india["Cured"],color="b")

plt.xlabel("Tarih")
plt.ylabel("İyileşen Vakalar")

plt.xticks(rotation=75)
plt.show()

plt.figure(figsize=(20,10))
plt.bar(df_india["State/UnionTerritory"],df_india["Confirmed"])
plt.xlabel("Eyaletler")
plt.ylabel("Onaylanmış Yerli Hint Vatandaşı + Onaylanmış Yabancı Uyruklu")
plt.xticks(rotation=75)
plt.show()
df_india_date_wise["Date"] = pd.to_datetime(df_india_date_wise["Date"],infer_datetime_format=True)
plt.figure(figsize=(30,20))
sns.lineplot(tmp["Date"],tmp["Cured"])
sns.lineplot(tmp["Date"],tmp["Deaths"])
plt.xlabel("Tarih")
plt.ylabel("Miktar")
plt.xticks(rotation=75)
plt.legend(["İyileşen Vakalar","Ölümle Sonuçlanan Vakalar"])
plt.show()
plt.figure(figsize=(20,10))
sns.lineplot(df_india["Date"],df_india["Deaths"])
# plt.bar(df_india["Date"],df_india["ConfirmedIndianNational"])
sns.lineplot(df_india["Date"],df_india["Cured"])
plt.legend(["Ölümle Sonuçlanan Vakalar","İyileşen Vakalar"])
plt.xticks(rotation=75)
plt.show()
plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
plt.bar(df_india["Date"],df_india["Deaths"],color='coral')
# plt.bar(df_india["Date"],df_india["ConfirmedIndianNational"])
plt.xticks(rotation=75)
plt.ylim(0, 3000)
plt.xlabel("Tarih")
plt.ylabel("Ölümle Sonuçlanan Vakalar")

plt.subplot(2,1,2)
plt.bar(df_india["Date"],df_india["Cured"],color='darkolivegreen')
plt.xlabel("Tarih")
plt.ylabel("İyileşen Vakalar")
plt.ylim(0, 35000)

plt.xticks(rotation=75)
plt.show()
plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
plt.bar(df_india["State/UnionTerritory"],df_india["Deaths"],color='coral')
# plt.bar(df_india["Date"],df_india["ConfirmedIndianNational"])
plt.xticks(rotation=75)
plt.ylim(0, 3000)
plt.xlabel("Eyalet")
plt.ylabel("Ölümle Sonuçlanan Vakalar")

plt.subplot(2,1,2)
plt.bar(df_india["State/UnionTerritory"],df_india["Cured"],color='darkolivegreen')
plt.xlabel("Eyaletler")
plt.ylabel("İyileşen Vakalar")
plt.ylim(0, 35000)

plt.xticks(rotation=75)
plt.show()
df_individual = pd.read_csv("/kaggle/input/covid19-in-india/IndividualDetails.csv")
df_individual.head()
df_individual['gender']=df_individual['gender'].fillna("UNK")
df_individual["age"] = df_individual["age"].fillna(0)
df_individual["age"] = df_individual['age'].replace('28-35',32)
df_individual['diagnosed_date']=pd.to_datetime(df_individual['diagnosed_date'],dayfirst=True,infer_datetime_format=True)
df_individual['current_status'].value_counts()
df_individual.isnull().sum()
df_individual[df_individual['detected_state'].isnull()]
df_individual['gender'].value_counts()
df_individual_tmp = df_individual
df_individual_tmp.head()
df_individual_tmp["age"].value_counts()
df_individual_tmp.info()
df_individual_tmp['diagnosed_date']=pd.to_datetime(df_individual_tmp['diagnosed_date'],dayfirst=True,infer_datetime_format=True)
df_individual_tmp['current_status'].value_counts()
deaths_state=df_individual_tmp['detected_state'][df_individual_tmp['current_status']=='Deceased'].value_counts()
deaths_state=dict(deaths_state)
df_age_group.head(20)
df_age_group.info()
from matplotlib.pyplot import pie, axis
plt.figure(figsize=(20,10))
sums = df_age_group.groupby(df_age_group["Percentage"])["TotalCases"].sum()
axis('equal');
pie(sums, labels=sums.index);

plt.show()
df_imcrt.head()
df_imcrt.info()
df_population.shape
df_hospital_bed.head()
df_hospital_bed.shape
recent_date = df_india['Date'].max()
mortality_statewise = df_india[df_india['Date']==recent_date]
mortality_statewise.head()
mortality_statewise.shape
mortality_statewise['Recovery_Rate']=(mortality_statewise['Cured']/mortality_statewise['Total Cases'])*100
mortality_statewise.head()
covid_19_india_color=covid_19_india[["Date",'State/UnionTerritory','Confirmed','Cured','Deaths']]
covid_19_india_color=covid_19_india_color.sort_values(['Confirmed'],ascending=False)
covid_19_india_color.style.background_gradient(cmap='Reds')
#Tahmin yapacağımız veri seti
df_india = pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv")
#Ölüm sayılarını görmek için 2000 dedim
df_india.head(2000)
#Mevcut columnları görmek için kullanıcaz aşağıdaki kodu yazıyoruz.
df_india.columns
#işimize yarayan kolonları alıyoruz
selected_features=['Sno','Cured','Deaths','Confirmed']
X=df_india[selected_features]
y=df_india.Deaths
y.describe()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
from sklearn.metrics import mean_absolute_error
olum_tahmin_oranı=regressor.predict(X_test)
mean_absolute_error(y_test,olum_tahmin_oranı)
#Tahmin yapacağımız veri seti
import pandas as pd
dataset = pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv")
#verilerimizi sayısal veriye dönüştürüyoruz.
dataset=pd.get_dummies(dataset)
dataset.columns
dataset.head(2000)
dataset.shape
#toplam hasta,sayısı,hastalık artış miktarını alıyoruz
X = dataset.iloc[:, [1,4]].values
#ölümleri tahmim etmeye çalışıyoruz
y = dataset.iloc[:, 3].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#from matplotlib.colors import ListedColormap
#X_set, y_set = X_train, y_train
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('blue', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('yellow', 'green'))(i), label = j)
#plt.title('Lojistik Regresyon (Eğitim seti)')
#plt.xlabel('Hasta Sayısı')
#plt.ylabel('Onaylanan Vaka Sayısı')
#plt.legend()
#plt.show()