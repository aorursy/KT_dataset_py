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
df_train = pd.read_csv('/kaggle/input/sanbercode-data-science-0620/train.csv')

df_test = pd.read_csv('/kaggle/input/sanbercode-data-science-0620/test.csv')

df_train.tail()
df_train.shape
df_train.info()
unique_val = len(set(df_train['id']))

total = df_train.shape[0]

print(f"Ada {total - unique_val} Id yang tidak unik.")
df_train.isna().sum()
df_train.describe()
#statistika deskriptif untuk kategorikal feature

df_train.describe(include = np.object)
df_train['Gaji'] = df_train['Gaji'].replace(['<=7jt', '>7jt'],[0, 1])

df_train.head()
cat_features = df_train.dtypes[(df_train.dtypes == 'object')].index

num_features = df_train.dtypes[(df_train.dtypes != 'object')].index
for i in range(len(cat_features)):

    print(f"{cat_features[i]}: {df_train[cat_features[i]].unique()}")
df_train = df_train.replace('?', np.nan)

df_test = df_test.replace('?', np.nan)



df_train = df_train.fillna(method='ffill')

df_test = df_test.fillna(method='ffill')



# mengganti Beripsah dan Janda dengan Cerai

df_train['Status Perkawinan'] = df_train['Status Perkawinan'].replace(['Berpisah', 'Janda'], 'Cerai')

df_test['Status Perkawinan'] = df_test['Status Perkawinan'].replace(['Berpisah', 'Janda'], 'Cerai')



#mengganti Menikah LDR dengan Menikah

df_train['Status Perkawinan'] = df_train['Status Perkawinan'].replace('Menikah LDR', 'Menikah')

df_test['Status Perkawinan'] = df_test['Status Perkawinan'].replace('Menikah LDR', 'Menikah')



for i in range(len(cat_features)):

    print(f"{cat_features[i]}: {df_train[cat_features[i]].unique()}")
i=7

df_train[df_train['Kelas Pekerja']==i]['Gaji'].value_counts()/df_train[df_train['Kelas Pekerja']==i].shape[0]
col1 = {'SD':0,'1st-4th':1, '5th-6th':2, '7th-8th':3, 

        '9th':4, '10th':5, '11th':6, '12th':7, 'SMA':8, 

        'Pendidikan Tinggi':9, 'D4':10, 'D3':11, 'Sarjana':12, 

        'Master':13, 'Sekolah Professional':14, 'Doktor':15 }



df_train['Pendidikan'] = df_train['Pendidikan'].replace(col1)

df_test['Pendidikan'] = df_test['Pendidikan'].replace(col1)



col2 = {'Belum Pernah Menikah':0, 'Cerai':2, 'Menikah':1}



#df_train['Status Perkawinan'] = df_train['Status Perkawinan'].replace(col2)

#df_test['Status Perkawinan'] = df_test['Status Perkawinan'].replace(col2)



col3 = {'Servis Lainnya':0, 'Ekesekutif Managerial':1, 'Spesialis':2,

 'Perbaikan Kerajinan':7, 'Sales':6, 'Pembersih':5, 'Pemuka Agama':4, 'Petani':3,

 'Tech-support':8, 'Mesin Inspeksi':9, 'Supir':10, 'Asisten Rumah Tangga':11, 'Penjaga':12,

 'Tentara':13}



#df_train['Pekerjaan'] = df_train['Pekerjaan'].replace(col3)

#df_test['Pekerjaan'] = df_test['Pekerjaan'].replace(col3)



col4 = {'Wiraswasta':2, 'Pemerintah Lokal':5, 'Pekerja Bebas Perusahaan':3,

 'Pemerintah Negara':4, 'Pekerja Bebas Bukan Perusahan':7, 'Pemerintah Provinsi':6,

 'Tidak Pernah Bekerja':0, 'Tanpa di Bayar':1}



df_train['Kelas Pekerja'] = df_train['Kelas Pekerja'].replace(col4)

df_test['Kelas Pekerja'] = df_test['Kelas Pekerja'].replace(col4)
for i in range(len(cat_features)):

    print(f"{num_features[i]}: {df_train[num_features[i]].unique()}")
cat_features = df_train.dtypes[(df_train.dtypes == 'object')].index

num_features = df_train.dtypes[(df_train.dtypes != 'object')].index
import seaborn as sns

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2,2, figsize=(20,20))

axes = axes.flatten()



i = 0

for t in cat_features[0:]:

    df_train[t].value_counts().plot(kind='bar', ax=axes[i])

    i +=1
x_var = "Gaji"

fig, axes = plt.subplots(3,3, figsize=(20,10))

axes = axes.flatten()



i = 0

for t in num_features[1:-1]:

    ax = sns.boxplot(x=x_var, y=t, data=df_train, ax=axes[i])

    i +=1
num_features[1:-1]
from sklearn.preprocessing import Normalizer

scaler = Normalizer()

df_norm = scaler.fit_transform(df_train[num_features[1:-1]])

df_train[num_features[1:-1]] = pd.DataFrame(df_norm, columns = num_features[1:-1])



df_norm = scaler.transform(df_test[num_features[1:-1]])

df_test[num_features[1:-1]] = pd.DataFrame(df_norm, columns = num_features[1:-1])

df_test.head()
corr = df_train.corr()

fig, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr, vmax=1, annot=False,square=True);

plt.show()
df_train.corr()
df_join = pd.concat([df_train,df_test],axis=0)

df_join = df_join.drop(['id'],axis=1)
df_dummy_join = pd.get_dummies(df_join,drop_first=True)

df_dummy_join.tail()
corr = df_dummy_join.corr()

fig, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr, vmax=1, annot=False,square=True);

plt.show()
df_dummy_join.Gaji.isna()
df_dummy_join[df_dummy_join.Gaji.isna()==True]
df_dummy_test = df_dummy_join[df_dummy_join.Gaji.isna()==True]

df_dummy_train = df_dummy_join[df_dummy_join.Gaji.isna()==False]



X = df_dummy_train.drop(['Gaji'],axis=1)

y = df_dummy_train['Gaji']

df_dummy_test.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 13, test_size=0.65, shuffle=True)
from imblearn.over_sampling import SMOTE



oversample = SMOTE()

X_train, y_train = oversample.fit_resample(X_train, y_train)
y_train.value_counts().plot(kind='bar')
from sklearn.metrics import classification_report
import tensorflow as tf

from tensorflow import keras

y_train_tf = tf.keras.utils.to_categorical(y_train.map({0:0,1:1}), num_classes=2)

y_test_tf = tf.keras.utils.to_categorical(y_test.map({0:0,1:1}), num_classes=2)
y_train_tf.shape
from keras.models import Sequential

from keras.layers import Dense



model1 = Sequential()

model1.add(Dense(128, input_shape=(X_train.shape[1],), activation="relu"))

model1.add(Dense(64, activation="relu"))

#model.add(Dense(64, activation="relu"))



model1.add(Dense(y_train_tf.shape[1]))



model1.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])



#model.fit(X_train, y_train_tf, epochs=10, batch_size=32)
model1.fit(X_train, y_train_tf, epochs=50, batch_size=128)
ypred=model1.predict(X_test)
ypred = [np.argmax(i) for i in ypred]

yhat = pd.Series(ypred).map({0:0,1:1})
print(classification_report(y_test, yhat))
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier()

random_forest.fit(X_train, y_train)

yhat = random_forest.predict(X_test)

print(f"Train accuracy: {random_forest.score(X_train, y_train)}")

print(f"Test accuracy: {random_forest.score(X_test, y_test)}")
from sklearn.metrics import roc_auc_score

ypred = random_forest.predict(X_test)

roc_auc_score(y_test,ypred)
from sklearn.metrics import roc_curve



ypred = random_forest.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, ypred)
plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='Knn')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('Neural Networks')



plt.show()
from sklearn import metrics

roc_auc = metrics.auc(fpr, tpr)

print("roc_auc:  %0.2f" % roc_auc)
XTest = df_dummy_test.drop(['Gaji'],axis=1)

yTest = df_dummy_test['Gaji']
ypred = model1.predict(XTest)
ypred = [np.argmax(i) for i in ypred]

yhat = pd.Series(ypred).map({0:0,1:1})
yhat
result = pd.DataFrame({'id':np.array(df_test['id'].values), 'Gaji': yhat})

result.to_csv('result', index=False)

result.head()
df_test = pd.read_csv('/kaggle/input/sanbercode-data-science-0620/test.csv')

data = df_test.merge(result, on="id")

data.tail(10)