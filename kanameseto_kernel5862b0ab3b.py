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
df=pd.read_csv("/kaggle/input/1056lab-used-cars-price-prediction/train.csv",index_col=0)
df
df['Name'].value_counts()
df['Location'].value_counts()
df['New_Price'].value_counts()
df.isnull().any()
dummy = pd.get_dummies(df[['Location','Fuel_Type','Transmission','Owner_Type']], drop_first=True)

dummy
dfe = pd.concat([df, dummy], axis=1)

dfe=dfe.drop(['Location','Fuel_Type','Transmission','Owner_Type','New_Price','Name'],axis=1)
dfe
dfe['Mileage']=dfe['Mileage'].str.replace('kmpl','')

dfe['Mileage']=dfe['Mileage'].str.replace('km/kg','')

dfe['Engine']=dfe['Engine'].str.replace('CC','')

dfe['Power']=dfe['Power'].str.replace('bhp','')

dfe['Mileage']=dfe['Mileage'].str.replace('null','0')

dfe['Engine']=dfe['Engine'].str.replace('null','0')

dfe['Power']=dfe['Power'].str.replace('null','0')
dfe['Engine']=dfe['Engine'].astype(float)

dfe['Power']=dfe['Power'].astype(float)

dfe['Mileage']=dfe['Mileage'].astype(float)



dfe.loc[dfe.Engine==0,'Engine']=np.NaN

dfe.loc[dfe.Power==0,'Power']=np.NaN





dfe['Engine']=dfe['Engine'].fillna(dfe['Engine'].mean())

dfe['Power']=dfe['Power'].fillna(dfe['Power'].mean())

dfe['Seats']=dfe['Seats'].fillna(dfe['Seats'].mean())
from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(n_estimators=100,random_state=72)
X=dfe.drop(['Price'],axis=1).values

y=dfe.Price.values
rfr.fit(X,y)
dft=pd.read_csv('/kaggle/input/1056lab-used-cars-price-prediction/test.csv',index_col=0)
dummy = pd.get_dummies(dft[['Location','Fuel_Type','Transmission','Owner_Type']], drop_first=True)
dfet = pd.concat([dft, dummy], axis=1)

dfet=dfet.drop(['Location','Fuel_Type','Transmission','Owner_Type','New_Price','Name'],axis=1)
dfet['Mileage']=dfet['Mileage'].str.replace('kmpl','')

dfet['Mileage']=dfet['Mileage'].str.replace('km/kg','')

dfet['Engine']=dfet['Engine'].str.replace('CC','')

dfet['Power']=dfet['Power'].str.replace('bhp','')

dfet['Mileage']=dfet['Mileage'].str.replace('null','0')

dfet['Engine']=dfet['Engine'].str.replace('null','0')

dfet['Power']=dfet['Power'].str.replace('null','0')
dfet['Engine']=dfet['Engine'].astype(float)

dfet['Power']=dfet['Power'].astype(float)

dfet['Mileage']=dfet['Mileage'].astype(float)



dfet.loc[dfet.Engine==0,'Engine']=np.NaN

dfet.loc[dfet.Power==0,'Power']=np.NaN





dfet['Engine']=dfet['Engine'].fillna(dfet['Engine'].mean())

dfet['Power']=dfet['Power'].fillna(dfet['Power'].mean())

dfet['Seats']=dfet['Seats'].fillna(dfet['Seats'].mean())

dfet['Mileage']=dfet['Mileage'].fillna(dfet['Mileage'].mean())
dfet=dfet.drop('Fuel_Type_Electric',axis=1)
Xt=dfet.values
predict=rfr.predict(Xt)
submit = pd.read_csv('/kaggle/input/1056lab-used-cars-price-prediction/sampleSubmission.csv')

submit['Price'] = predict

submit.to_csv('submission.csv', index=False)