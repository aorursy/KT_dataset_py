

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.model_selection import train_test_split 

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder



from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectFromModel

from sklearn import metrics

from sklearn.metrics import mean_squared_error,r2_score





import os

print(os.listdir("../input"))
def load_datasets():

    print("in load_datasets() funciton")

    temp_datasets=dict()

    for file_name in os.listdir("../input"):

        dataset_name=file_name.split("PM")[0].lower()

        dataset=pd.read_csv("../input/"+file_name, 

                        parse_dates={'dt' : ['year', 'month','day','hour']}, 

                        date_parser=lambda x: pd.datetime.strptime(x, '%Y %m %d %H'),

                        infer_datetime_format=True,

                        index_col='dt',

                       na_values=['NaN','?'],)

        temp_datasets[dataset_name]=dataset

    

    return temp_datasets
def mydropna(mydatasets):

    print("in mydropna() funciton")

    mytemp_datasets=dict()

    for city, dataset in mydatasets.items():

        mytemp_datasets[city]=dataset.dropna(axis=0, how="any")

        print("eksiltmeden önce ",city," dataset shape:",dataset.shape)

        print("eksiltmeden sonra ",city," dataset shape:",mytemp_datasets[city].shape)

        print()

    

    return mytemp_datasets
datasets=load_datasets()
beijing=datasets['beijing'].copy()
beijing.drop(['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan'], 

               axis=1,

              inplace=True)
beijing.head()
beijing.dropna(axis=0, how="any",inplace=True)


labelEncoder=LabelEncoder()

beijing['cbwd']=labelEncoder.fit_transform(beijing['cbwd'])
X=beijing.drop('PM_US Post', axis=1)

y=beijing['PM_US Post']
standardScaler=StandardScaler()

X_scaled=standardScaler.fit_transform(X)
print("X_scaled.shape:",X_scaled.shape)

print("y.shape:",y.shape)
train_size=int(y.shape[0]*0.8)

test_size=y.shape[0]-train_size

print("train size:",train_size)

print("test size :",test_size)
X_train=X_scaled[:train_size]

y_train=y[:train_size]

X_test=X_scaled[train_size:]

y_test=y[train_size:]
print("X_train.shape:",X_train.shape)

print("y_train.shape:",y_train.shape)

print("X_test.shape:",X_test.shape)

print("y_test.shape",y_test.shape)
from sklearn.linear_model import LinearRegression

linearRegression=LinearRegression()

linearRegression.fit(X_train,y_train)
r2=linearRegression.score(X_test,y_test)

print("R2 skoru:{:.4f}".format(r2))
y_pred=linearRegression.predict(X_test)
print("Ortalama Kare Hatası:{:.4f}".format(mean_squared_error(y_test, y_pred)))
print("R2 skoru:{:.4f}".format(r2_score(y_test,y_pred)))
datasets_na_dropped=mydropna(datasets)
for dataset_name, dataset in datasets_na_dropped.items():

    print(dataset_name.upper()," şehrinde yer alan ölçüm istasyonlar:")

    for column in dataset.columns.values:

        if "PM_" in column:

            print(column)

    print()
for dataset_name, dataset in datasets_na_dropped.items():

    print(dataset_name," veri setine ait ilk beş satır:")

    print(dataset.head())


for city, dataset in datasets_na_dropped.items():

    columns=[]

    for column in dataset.columns.values:

        if "PM_" in column:

            columns.append(column)

    datasets_na_dropped[city][columns][::360].plot(figsize=(20,4),title="City: "+city)



plt.show()
datasets=load_datasets()

for city, dataset in datasets.items():

    columns=list()

    for column in dataset.columns.values:

        if 'PM' in column:

            columns.append(column)

    print(city," şehri için ölçüm istasyonlarının ölçüm sayıları:")

    print(dataset[columns].notnull().sum())
datasets['beijing'].head()
datasets_only_USPostPM={}

for city, dataset in datasets.items():

    columns=list()

    for column in dataset.columns.values:

        if 'PM' in column and "US Post" not in column:

            columns.append(column)

    # No başlıklı sütun gereksiz olduğu için çıkartılacak sütünlar listesine eklenir

    columns.append('No')

    datasets_only_USPostPM[city]=dataset.drop(columns=columns)
datasets_only_USPostPM['beijing'].head()
for city, dataset in datasets_only_USPostPM.items():

    dataset.dropna(axis="index", inplace=True, how="any")
datasets_only_USPostPM['beijing'].head()
datasets_only_USPostPM['beijing'].info()
labelEncoder=LabelEncoder()

for city, dataset in datasets_only_USPostPM.items():

    labelEncoder.fit(dataset['cbwd'])

    dataset['cbwd']=labelEncoder.transform(dataset['cbwd'])

    dataset=pd.concat([dataset,pd.get_dummies(dataset['cbwd'], prefix="cbwd")],axis=1)

    dataset.drop(['cbwd'],inplace=True, axis=1)

    datasets_only_USPostPM[city]=dataset
from sklearn.linear_model import LinearRegression

linearRegression=LinearRegression()
print(len(datasets_only_USPostPM['beijing']))
X=datasets_only_USPostPM['beijing'].drop('PM_US Post', axis=1)

y=datasets_only_USPostPM['beijing']['PM_US Post']
train_size=int(len(datasets_only_USPostPM['beijing'])*0.8)

test_size=len(datasets_only_USPostPM['beijing'])-train_size

print("eğitim örnek sayısı:",train_size)

print("test örnek sayısı:",test_size)

print("toplam:",train_size+test_size)
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler(copy="deep")

scaler.fit(X)

X=scaler.transform(X)
X_train=X[:train_size]

X_test=X[train_size:]

y_train=y[:train_size]

y_test=y[train_size:]
linearRegression.fit(X_train,y_train)

y_pred=linearRegression.predict(X_test)

linearRegression.score(X_test, y_test)
n_results=100

fig, ax=plt.subplots(2,1,figsize=(12,8))

ax[0].plot(y_test.values[:n_results], color="red")

ax[1].plot(y_pred[:n_results], color="green")
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))