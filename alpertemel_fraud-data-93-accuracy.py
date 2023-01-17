# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings
df=pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
df.head()
df.describe().T
df.isnull().sum()/len(df)
corr=df.corr()

sns.heatmap(corr)
df["Class"].head()
df["Class"].value_counts()
sns.countplot(df.Class)
fraud=df.loc[df["Class"]==1]



fraud
i=0

fraud=df.iloc[0:1,:]



while i<284807:

    

    if df.Class[i]==1:

        fraud=pd.concat([fraud,df.iloc[i:i+1,:]],axis=0)

        

    i+=1

    

fraud=fraud.iloc[1:,:]

not_fraud=df.loc[df["Class"]==0]

not_fraud
fraud["Amount"].value_counts(1)
not_fraud["Amount"].value_counts(1)
sns.distplot(fraud.Time)
sns.distplot(not_fraud.Time)
fraud.describe()
not_fraud.describe()
yeni_not_fraud=not_fraud.iloc[0:1,:]

yeni_not_fraud
import random
while len(yeni_not_fraud)<492:

    

    i=random.randrange(0,284315)

    yeni_not_fraud=pd.concat([not_fraud.iloc[i:i+1,:],yeni_not_fraud],axis=0)

    
yeni_not_fraud
yeni_df=pd.concat([fraud,yeni_not_fraud],axis=0)

yeni_df
corr_yeni=yeni_df.corr()

sns.heatmap(corr_yeni)
sns.heatmap(corr)
from sklearn.preprocessing import StandardScaler, RobustScaler



std_scaler = StandardScaler()

rob_scaler = RobustScaler()



yeni_df['scaled_amount'] = rob_scaler.fit_transform(yeni_df['Amount'].values.reshape(-1,1))

yeni_df['scaled_time'] = rob_scaler.fit_transform(yeni_df['Time'].values.reshape(-1,1))



yeni_df.drop(['Time','Amount'], axis=1, inplace=True)







scaled_amount = yeni_df['scaled_amount']

scaled_time = yeni_df['scaled_time']



yeni_df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)

yeni_df.insert(0, 'scaled_amount', scaled_amount)

yeni_df.insert(1, 'scaled_time', scaled_time)

yeni_df
x=yeni_df.iloc[:,0:30]

y=yeni_df.iloc[:,30:31]
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
from sklearn.ensemble import RandomForestClassifier



rf=RandomForestClassifier()

rf.fit(x_train,y_train)

rf_tahmin=rf.predict(x_test)
from xgboost import XGBClassifier



xgb=XGBClassifier()

xgb.fit(x_train,y_train)

xgb_tahmin=xgb.predict(x_test)

from lightgbm import LGBMClassifier



lgb=LGBMClassifier()

lgb.fit(x_train,y_train)

lgb_tahmin=lgb.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix



print(accuracy_score(y_test,rf_tahmin),accuracy_score(y_test,xgb_tahmin),accuracy_score(y_test,lgb_tahmin))
X=df.iloc[:,0:30]

Y=df.iloc[:,30:31]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=34)
rf2=RandomForestClassifier()

rf2.fit(X_train,Y_train)

rf2_tahmin=rf2.predict(X_test)

accuracy_score(Y_test,rf2_tahmin)