# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



train = pd.read_csv('/kaggle/input/sanbercode-data-science-0620/train.csv')

test = pd.read_csv('/kaggle/input/sanbercode-data-science-0620/test.csv')

sample= pd.read_csv('/kaggle/input/sanbercode-data-science-0620/sampleSubmission.csv')

dede= train.loc[train['Kelas Pekerja']!='?'].copy()

List_Pendidikan = list(dede['Pendidikan'].unique())

dicts={}

dede.loc[dede['Pendidikan']=='SMA','Kelas Pekerja'].mode()

for i in List_Pendidikan:

    dicts[i]= dede.loc[dede['Pendidikan']==i,'Kelas Pekerja'].mode().values[0]

dicts

train['Kelas Pekerja'] = train['Kelas Pekerja'].replace({'?': 'Wiraswasta'})

test['Kelas Pekerja'] = test['Kelas Pekerja'].replace({'?': 'Wiraswasta'})

dicts_pekerjaan= {}

for i in List_Pendidikan:

    dicts_pekerjaan[i] = dede.loc[dede['Pendidikan']==i,'Pekerjaan'].mode().values[0]

print(dicts_pekerjaan)
train.loc[train['Kelas Pekerja']=='?'].index.tolist()

train['Pekerjaan'].value_counts()
train
berat_label= [1,2,3,4,5,6,7,8,9,10]

test['Berat Akhir']=pd.qcut(test['Berat Akhir'],10, labels=berat_label)

train['Berat Akhir']=pd.qcut(train['Berat Akhir'],10, labels=berat_label)
train['Pendidikan']=train['Pendidikan'].replace({'SMA':1,'10th':2,'D3':3,'Sarjana':4,'Master':5,'Pendidikan Tinggi':6,'1st-4th':7,'Sekolah Professional':8,'7th-8th':9,'Doktor':10,'5th-6th':11,'11th':12,'9th':13,'D4':14, '12th':15,'SD':16})

test['Pendidikan']=test['Pendidikan'].replace({'SMA':1,'10th':2,'D3':3,'Sarjana':4,'Master':5,'Pendidikan Tinggi':6,'1st-4th':7,'Sekolah Professional':8,'7th-8th':9,'Doktor':10,'5th-6th':11,'11th':12,'9th':13,'D4':14, '12th':15,'SD':16})

train['Gaji']=train['Gaji'].replace({'<=7jt':0,'>7jt':1})
for col in train:

    if col=='Gaji':

        break

    else:

        print(train[[col,'Gaji']].groupby([col],as_index=False).mean())
train['Status Perkawinan'] = train['Status Perkawinan'].replace({'Menikah':5, 'Belum Pernah Menikah':4,'Cerai':3, 'Berpisah':2, 'Janda':1, 'Menikah LDR':0})

test['Status Perkawinan'] = test['Status Perkawinan'].replace({'Menikah':5, 'Belum Pernah Menikah':4,'Cerai':3, 'Berpisah':2, 'Janda':1, 'Menikah LDR':0})
train['Kelas Pekerja'].unique()

train['Kelas Pekerja']=train['Kelas Pekerja'].replace({'Wiraswasta':1, 'Pemerintah Lokal':2, 'Pekerja Bebas Perusahaan':3,

       'Pemerintah Negara':4, 'Pekerja Bebas Bukan Perusahan':5,

       'Pemerintah Provinsi':6, 'Tidak Pernah Bekerja':7, 'Tanpa di Bayar':8})

test['Kelas Pekerja']=test['Kelas Pekerja'].replace({'Wiraswasta':1, 'Pemerintah Lokal':2, 'Pekerja Bebas Perusahaan':3,

       'Pemerintah Negara':4, 'Pekerja Bebas Bukan Perusahan':5,

       'Pemerintah Provinsi':6, 'Tidak Pernah Bekerja':7, 'Tanpa di Bayar':8})
train['Pekerjaan']=train['Pekerjaan'].replace({'Servis Lainnya':1, 'Ekesekutif Managerial':2, 'Spesialis':3,

       'Perbaikan Kerajinan':4, '?':5, 'Sales':6, 'Pembersih':7, 'Pemuka Agama':8,

       'Petani':9, 'Tech-support':10, 'Mesin Inspeksi':11, 'Supir':12,

       'Asisten Rumah Tangga':13, 'Penjaga':14, 'Tentara':15})

test['Pekerjaan']=test['Pekerjaan'].replace({'Servis Lainnya':1, 'Ekesekutif Managerial':2, 'Spesialis':3,

       'Perbaikan Kerajinan':4, '?':5, 'Sales':6, 'Pembersih':7, 'Pemuka Agama':8,

       'Petani':9, 'Tech-support':10, 'Mesin Inspeksi':11, 'Supir':12,

       'Asisten Rumah Tangga':13, 'Penjaga':14, 'Tentara':15})
train['Jenis Kelamin'] =train['Jenis Kelamin'].replace({'Perempuan':1,'Laki2':2})

test['Jenis Kelamin'] =test['Jenis Kelamin'].replace({'Perempuan':1,'Laki2':2})
a, b  =pd.cut(train['Keuntungan Kapital'],10 ,labels=[1,2,3,4,5,6,7,8,9,10], retbins=True)

train['Keuntungan Kapital']= pd.cut(train['Keuntungan Kapital'], bins=b,labels=[1,2,3,4,5,6,7,8,9,10])

test['Keuntungan Kapital']= pd.cut(test['Keuntungan Kapital'], bins=b,labels=[1,2,3,4,5,6,7,8,9,10])



#Memberikan pembagian label pada kerugian capital

c, d = pd.cut(train['Kerugian Capital'],5 ,labels=[1,2,3,4,5], retbins=True)

train['Kerugian Capital']= pd.cut(train['Kerugian Capital'], bins=d,labels=[1,2,3,4,5])

test['Kerugian Capital']= pd.cut(test['Kerugian Capital'], bins=d,labels=[1,2,3,4,5])
test.isnull().sum()
import matplotlib.pyplot as plt
plt.boxplot(train['Jam per Minggu'])
e,f = pd.cut(train['Jam per Minggu'],3, labels=[1,2,3],retbins=True)

train['Jam per Minggu']=pd.cut(train['Jam per Minggu'],bins=f, labels=[1,2,3])

test['Jam per Minggu']=pd.cut(test['Jam per Minggu'],bins=f, labels=[1,2,3])
plt.boxplot(train['Umur'])

print(train['Umur'].describe())
g,h = pd.cut(train['Umur'],3, labels=[1,2,3],retbins=True)

train['Umur']=pd.cut(train['Umur'],bins=h, labels=[1,2,3])

test['Umur']=pd.cut(test['Umur'],bins=h, labels=[1,2,3])
test.head()

train.head()
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score
X_train = train.drop(["id","Gaji"], axis=1)

Y_train = train["Gaji"]

X_test  = test.drop("id", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred_LS = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
Y_train_pred_LS = logreg.predict(X_train)

roc_auc_score(Y_train,Y_train_pred_LS)
random_forest = RandomForestClassifier(n_estimators=200)

random_forest.fit(X_train, Y_train)

Y_pred_RF = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
Y_train_RF = random_forest.predict(X_train)

roc_auc_score(Y_train,Y_train_RF)
knn = KNeighborsClassifier(n_neighbors = 24, weights='uniform')

knn.fit(X_train, Y_train)

Y_pred_KNN = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
Y_train_KNN = random_forest.predict(X_train)

roc_auc_score(Y_train,Y_train_KNN)
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV

import numpy as np

model = KNeighborsClassifier()

param_grid ={'n_neighbors':np.arange(5,25), 'weights':['uniform','distance']}

gscv = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=5)

gscv.fit(X_train, Y_train)
gscv.best_params_
test['Gaji'] = Y_pred_RF

test.head()
train2 = pd.read_csv('/kaggle/input/sanbercode-data-science-0620/train.csv')

test2 = pd.read_csv('/kaggle/input/sanbercode-data-science-0620/test.csv')
train['Kelas Pekerja'] = train2['Kelas Pekerja'].copy()

test['Kelas Pekerja'] = test2['Kelas Pekerja'].copy()

train['Pekerjaan'] = train2['Pekerjaan'].copy()

test['Pekerjaan'] = test2['Pekerjaan'].copy()
test.head()
train.head()
gabungan = pd.concat([train,test],ignore_index=False)

gabungan['Pekerjaan']=gabungan['Pekerjaan'].replace({'Servis Lainnya':1, 'Ekesekutif Managerial':2, 'Spesialis':3, 'Perbaikan Kerajinan':4,  'Sales':5, 'Pembersih':6, 'Pemuka Agama':7, 'Petani':8, 'Tech-support':9, 'Mesin Inspeksi':10, 'Supir':11, 'Asisten Rumah Tangga':12, 'Penjaga':13, 'Tentara':14})

gabungan['Kelas Pekerja']=gabungan['Kelas Pekerja'].replace({'Wiraswasta':1, 'Pemerintah Lokal':2, 'Pekerja Bebas Perusahaan':3,

       'Pemerintah Negara':4, 'Pekerja Bebas Bukan Perusahan':5,

       'Pemerintah Provinsi':6, 'Tidak Pernah Bekerja':7, 'Tanpa di Bayar':8})
gabungan.tail()
data1 = gabungan.loc[gabungan['Kelas Pekerja']!='?'].copy()

data2 = gabungan.loc[gabungan['Kelas Pekerja']=='?'].copy()
data1['Pekerjaan']
data1['Pekerjaan'].value_counts()

data1['Pekerjaan']= data1['Pekerjaan'].replace({'?':data1['Pekerjaan'].mode().values[0]})
x_data1_train = data1.drop(['id','Kelas Pekerja','Pekerjaan'], axis=1).copy()

y_data1_train_KP = data1['Kelas Pekerja'].copy().astype('int64')

x_data2_test = data2.drop(['id','Kelas Pekerja','Pekerjaan'], axis=1).copy()
x_data1_train
x_data2_test 
model_RFC = RandomForestClassifier(n_estimators=100)

model_RFC.fit(x_data1_train, y_data1_train_KP)

y_data1_test = model_RFC.predict(x_data2_test)

acc_model_RFC = round(model_RFC.score(x_data1_train, y_data1_train_KP) * 100, 2)

acc_model_RFC
data2['Kelas Pekerja'] = y_data1_test

data2['Kelas Pekerja'].value_counts()
data1['Kelas Pekerja'] = data1['Kelas Pekerja'].astype('int64')
data1['Kelas Pekerja'] 
x_data1_train_P = data1.drop(['id','Pekerjaan'], axis=1).copy()

y_data1_train_P = data1['Pekerjaan'].copy()

x_data2_test_P = data2.drop(['id','Pekerjaan'], axis=1).copy()
x_data2_test_P
y_data1_train_P.value_counts()
model_RFC = RandomForestClassifier(n_estimators=100)

model_RFC.fit(x_data1_train_P, y_data1_train_P)

y_data2_test = model_RFC.predict(x_data2_test_P)

acc_model_RFC = round(model_RFC.score(x_data1_train_P, y_data1_train_P) * 100, 2)

acc_model_RFC
y_data2_test.shape
data2['Pekerjaan'] = y_data2_test

data2['Pekerjaan'].value_counts()
data1['Kelas Pekerja']
data1
test.head()
Data_lengkap  =Data_lengkap.sort_values(by=['id']).set_index('id')
Train_final = Data_lengkap.iloc[:35994]

Test_final = Data_lengkap.loc[35994:].drop('Gaji', axis=1)
Train_final
Test_final
x_Train_final = Train_final.drop('Gaji',axis=1).copy()

y_Train_final = Train_final['Gaji'].copy()

x_Test_final = Test_final.copy()
x_Train_final
RFC = RandomForestClassifier(n_estimators=250)

RFC.fit(x_Train_final, y_Train_final)

y_Test_final = RFC.predict(x_Test_final)

acc_RFC = round(RFC.score(x_Train_final, y_Train_final) * 100, 2)

acc_RFC
y_Train_final_Coba= RFC.predict(x_Train_final)

roc_auc_score(y_Train_final,y_Train_final_Coba)
knn = KNeighborsClassifier()

param_grid ={'n_neighbors':np.arange(5,50), 'weights':['uniform']}

gscv = GridSearchCV(knn, param_grid=param_grid, scoring='accuracy', cv=5)

gscv.fit(x_Train_final, y_Train_final)
gscv.best_params_
knn = KNeighborsClassifier(n_neighbors = 15, weights='uniform')

knn.fit(x_Train_final, y_Train_final)

y_Test_final_knn = knn.predict(x_Test_final)

acc_knn = round(knn.score(x_Train_final, y_Train_final) * 100, 2)

acc_knn
submission = pd.DataFrame({"id": test["id"],"Gaji": y_Test_final}) 

submission.to_csv('erickwang18_submission_5.csv',index=False)
s = pd.read_csv('./erickwang18_submission_1.csv')

s
a = submission['Gaji']

b = s['Gaji']

a.corr(b)