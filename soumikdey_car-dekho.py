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
df=pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/car data.csv')
df.shape
df.head()
df1=pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/CAR DETAILS FROM CAR DEKHO.csv')
df1.head()
df1.head()
df1.shape
df1.info()
df1.isnull().sum()
df1.duplicated().sum()
df1.drop_duplicates(inplace=True)
df1.shape
def process(car):
    return car.split()[0]
df1['brand']=df1['name'].apply(process)
df1.drop('name',inplace=True,axis=1)
df1.head()
df1['old']=2020-df1['year']
df1.head()
df1.drop('year',axis=1,inplace=True)
df1.head()
df1['fuel'].value_counts
df1=df1[~df1['fuel'].isin(['CNG','LPG','Electric'])]
df1['fuel'].value_counts()
df1.head()


df1=df1[~(df1['seller_type']=='Trustmark Dealer')]
df1['owner'].value_counts()
df1=df1[~df1['owner'].isin(['Fourth & Above Owner ','Test Drive Car'])]



df1.head()
from sklearn.preprocessing import LabelEncoder
fuel_encoder=LabelEncoder()
Seller_type_encoder=LabelEncoder()
Transmission_encoder=LabelEncoder()
Owner_encoder=LabelEncoder()
Brand_encoder=LabelEncoder()
df1['fuel']=fuel_encoder.fit_transform(df1['fuel'])
df1['seller_type']=Seller_type_encoder.fit_transform(df1['seller_type'])
df1['transmission']=Transmission_encoder.fit_transform(df1['transmission'])
df1['owner']=Owner_encoder.fit_transform(df1['owner'])
df1['brand']=Brand_encoder.fit_transform(df1['brand'])
df1.head()


from sklearn.preprocessing import OneHotEncoder
hot=OneHotEncoder(drop='first')
X_temp=hot.fit_transform(df1[['owner','brand']]).toarray()
X=df1.drop(['owner','brand','selling_price'],axis=1).values
X=np.hstack((X,X_temp))
y=df1.iloc[:,0].values
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=100)
rf_reg.fit(X_train,y_train)
y_pred=rf_reg.predict(X_test)
from sklearn.metrics import r2_score,mean_absolute_error
print(r2_score(y_pred,y_test))
import pickle
pickle.dump(fuel_encoder,open('fuel.pkl','wb'))
pickle.dump(Seller_type_encoder,open('seller.pkl','wb'))
pickle.dump(Transmission_encoder,open('Transmission.pkl','wb'))
pickle.dump(Owner_encoder,open('Owner.pkl','wb'))
pickle.dump(Brand_encoder,open('brand.pkl','wb'))
pickle.dump(hot,open('hot.pkl','wb'))
pickle.dump(rf_reg,open('rf.pkl','wb'))
pickle.dump(scaler,open('scaler.pkl','wb'))
df1.head()
