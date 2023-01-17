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
df=pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/train.csv',index_col=0,na_values='?')

df_t=pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/test.csv',index_col=0,na_values='?')
df
df_t
df.info()
df.isnull().sum()
df.symboling.value_counts()
df.make.value_counts()
categories =df['make'].unique()
df['make'] = pd.Categorical(df['make'], categories=categories)

dammies=pd.get_dummies(df.make)
dammies
dammies=dammies.drop(['mercedes-benz','bmw','plymouth','porsche','chevrolet','jaguar','isuzu','audi','saab','mercury','renault','alfa-romero'],axis=1)
df=pd.merge(df,dammies,on='id')
df['fuel-type'].value_counts()
df['fuel-type']=df['fuel-type'].map({'gas':1,'diesel':0})
df.aspiration.value_counts()
df['aspiration']=df['aspiration'].map({'std':1,'turbo':0})
df['num-of-doors'].value_counts()
df['num-of-doors']=df['num-of-doors'].map({'four':1,'two':0,np.NaN:2})
df['body-style'].value_counts()
df['body-style']=df['body-style'].map({'sedan':5,'hatchback':4,'wagon':3,'hardtop':2,'convertible':1})
df['drive-wheels'].value_counts()
df['drive-wheels']=df['drive-wheels'].map({'fwd':2,'rwd':1,'4wd':0})
df['engine-location'].value_counts()
df['engine-location']=df['engine-location'].map({'front':1,'rear':0})
df['engine-type'].value_counts()
df['engine-type']=df['engine-type'].map({'ohc':1})

df['engine-type']=df['engine-type'].fillna(0)
df['num-of-cylinders'].value_counts()
df['num-of-cylinders']=df['num-of-cylinders'].map({'four':1})

df['num-of-cylinders']=df['num-of-cylinders'].fillna(0)
df['fuel-system'].value_counts()
df['fuel-system']=df['fuel-system'].map({'mpfi':3,'2bbl':2,'idl':1})

df['fuel-system']=df['fuel-system'].fillna(0)
df['bore']=df['bore'].fillna(df['bore'].mean())

df['stroke']=df['stroke'].fillna(df['stroke'].mean())

df['horsepower']=df['horsepower'].fillna(df['horsepower'].mean())

df['peak-rpm']=df['peak-rpm'].fillna(df['peak-rpm'].mean())

df['price']=df['price'].fillna(df['price'].mean())
df=df.drop(['normalized-losses','make'],axis=1)
df
df_t['make'] = pd.Categorical(df_t['make'], categories=categories)

dammies=pd.get_dummies(df_t.make)

dammies=dammies.drop(['mercedes-benz','bmw','plymouth','porsche','chevrolet','jaguar','isuzu','audi','saab','mercury','renault','alfa-romero'],axis=1)

df_t=pd.merge(df_t,dammies,on='id')
df_t['fuel-type']=df_t['fuel-type'].map({'gas':1,'diesel':0})

df_t['aspiration']=df_t['aspiration'].map({'std':1,'turbo':0})

df_t['num-of-doors']=df_t['num-of-doors'].map({'four':1,'two':0,np.NaN:2})

df_t['body-style']=df_t['body-style'].map({'sedan':5,'hatchback':4,'wagon':3,'hardtop':2,'convertible':1})

df_t['drive-wheels']=df_t['drive-wheels'].map({'fwd':2,'rwd':1,'4wd':0})

df_t['engine-location']=df_t['engine-location'].map({'front':1,'rear':0})

df_t['engine-type']=df_t['engine-type'].map({'ohc':1})

df_t['engine-type']=df_t['engine-type'].fillna(0)

df_t['num-of-cylinders']=df_t['num-of-cylinders'].map({'four':1})

df_t['num-of-cylinders']=df_t['num-of-cylinders'].fillna(0)

df_t['fuel-system']=df_t['fuel-system'].map({'mpfi':3,'2bbl':2,'idl':1})

df_t['fuel-system']=df_t['fuel-system'].fillna(0)

df_t['bore']=df_t['bore'].fillna(df_t['bore'].mean())

df_t['stroke']=df_t['stroke'].fillna(df_t['stroke'].mean())

df_t['horsepower']=df_t['horsepower'].fillna(df_t['horsepower'].mean())

df_t['peak-rpm']=df_t['peak-rpm'].fillna(df_t['peak-rpm'].mean())

df_t['price']=df_t['price'].fillna(df_t['price'].mean())

df_t=df_t.drop(['normalized-losses','make'],axis=1)
X=df.drop(['symboling'],axis=1).values

y=df.symboling.values

Xt=df_t.values
#from imblearn.over_sampling import SMOTE

#sm = SMOTE()

#Xr, yr = sm.fit_sample(X, y)
#from sklearn.ensemble import RandomForestClassifier

#rfc=RandomForestClassifier(n_estimators=100,random_state=72)

#rfc.fit(X,y)
from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(n_estimators=100,random_state=72)

rfr.fit(X,y)
#predict = rfc.predict(Xt)

predict = rfr.predict(Xt)



submit = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/sampleSubmission.csv')

submit['symboling'] = predict

submit.to_csv('submission2.csv', index=False)