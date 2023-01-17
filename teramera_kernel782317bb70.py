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
train_data=pd.read_csv('/kaggle/input/DontGetKicked/training.csv')

train_data.head()
test_data=pd.read_csv('/kaggle/input/DontGetKicked/test.csv')

test_data.head()
test_data.shape
train_data.isnull().sum()
test_data.isnull().sum()
test_data["Transmission"].value_counts()
train_data["Transmission"].replace("Manual","MANUAL",inplace=True)
train_data["Color"].fillna(value="Color_unk",inplace=True)
test_data["Color"].fillna(value="Color_unk",inplace=True)
train_data["SubModel"].value_counts()
train_data["Transmission"].fillna(value="Transmission_unk",inplace=True)
test_data["Transmission"].fillna(value="Transmission_unk",inplace=True)
train_data["WheelType"].value_counts()
train_data["WheelType"].fillna(value="WheelType_unk",inplace=True)
test_data["WheelType"].fillna(value="WheelType_unk",inplace=True)
train_data["Nationality"].fillna(value="Nationality_unk",inplace=True)
test_data["Nationality"].fillna(value="Nationality_unk",inplace=True)
train_data["TopThreeAmericanName"].value_counts()
train_data["Size"].fillna(value="Size_unk",inplace=True)
test_data["Size"].fillna(value="Size_unk",inplace=True)
train_data["TopThreeAmericanName"].fillna(value="TopThreeAmericanName_unk",inplace=True)
test_data["TopThreeAmericanName"].fillna(value="TopThreeAmericanName_unk",inplace=True)
train_data["PRIMEUNIT"].fillna(value="PRIMEUNIT_unk",inplace=True)
test_data["PRIMEUNIT"].fillna(value="PRIMEUNIT_unk",inplace=True)
train_data["AUCGUART"].fillna(value="AUCGUART_unk",inplace=True)
test_data["AUCGUART"].fillna(value="AUCGUART_unk",inplace=True)
train_data.drop(["Model","Trim","SubModel","WheelTypeID","MMRAcquisitionAuctionAveragePrice","MMRAcquisitionAuctionCleanPrice","MMRAcquisitionRetailAveragePrice","MMRAcquisitonRetailCleanPrice","MMRCurrentAuctionAveragePrice","MMRCurrentAuctionCleanPrice","MMRCurrentRetailAveragePrice","MMRCurrentRetailCleanPrice","PurchDate"],axis=1,inplace=True)
test_data.drop(["Model","Trim","SubModel","WheelTypeID","MMRAcquisitionAuctionAveragePrice","MMRAcquisitionAuctionCleanPrice","MMRAcquisitionRetailAveragePrice","MMRAcquisitonRetailCleanPrice","MMRCurrentAuctionAveragePrice","MMRCurrentAuctionCleanPrice","MMRCurrentRetailAveragePrice","MMRCurrentRetailCleanPrice","PurchDate"],axis=1,inplace=True)
train_data.shape
test_data.shape
not_categorical=train_data.drop(["RefId","IsBadBuy"],axis=1).columns[train_data.drop(["RefId","IsBadBuy"],axis=1).dtypes!='object']
not_categorical
for i in not_categorical:

    maximum=np.max(train_data[i])

    train_data[i]=train_data[i]/maximum

    max_test=np.max(test_data[i])

    test_data[i]=test_data[i]/max_test
test_data.shape
test_data["VehicleAge"].value_counts()
test_data.columns
test_data.head()
categorical=train_data.drop(["RefId","IsBadBuy"],axis=1).columns[train_data.drop(["RefId","IsBadBuy"],axis=1).dtypes=='object']
categorical
pd.get_dummies(train_data[categorical[0]])
for i in categorical:

    dum=pd.get_dummies(train_data[i])

    dum.columns=str(i)+"_"+dum.columns

    train_data=pd.concat([train_data,dum],axis=1)

    train_data.drop(i,inplace=True,axis=1)

    dum=pd.get_dummies(test_data[i])

    dum.columns=str(i)+"_"+dum.columns

    test_data=pd.concat([test_data,dum],axis=1)

    test_data.drop(i,inplace=True,axis=1)
for i in train_data.drop("IsBadBuy",axis=1).columns:

    if i not in test_data.columns:

        test_data[i]=np.zeros(len(test_data))

for i in test_data.columns:

    if i not in train_data.columns:

        train_data[i]=np.zeros(len(train_data))

                
train_data.shape
test_data.shape
test_data=test_data[train_data.drop("IsBadBuy",axis=1).columns]
train_data.columns
test_data.columns
from sklearn.model_selection import train_test_split
x=train_data.drop(["RefId","IsBadBuy"],axis=1)

y=train_data["IsBadBuy"]
x_train,x_val,y_train,y_val=train_test_split(x,y,random_state=42)
x_train.shape

y_train.shape

x_val.shape

y_val.shape
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train,y_train)
knn.score(x_val,y_val)
pred=knn.predict(test_data.drop("RefId",axis=1))
submission=pd.DataFrame(data=pred,columns=["IsBadBuy"])
submission.head()
submission["RefId"]=test_data["RefId"]
submission.set_index("RefId",inplace=True)
submission.to_csv("submission.csv")
submission.head()