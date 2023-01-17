import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

from sklearn import preprocessing 



df=pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')
df.head()
df.info()
df.isnull().sum()
df.describe()
corr = df.iloc[:,:21].corr()

corr.style.background_gradient(cmap='coolwarm')
df.head(20)
df['Channel1'].value_counts()

df['TVConnection'].value_counts()



#df['TVConnection'].value_counts().count()
df['AddedServices'].value_counts()

df['HighSpeed'].value_counts()
df['gender']=df['gender'].astype('category').cat.codes

df['Married']=df['Married'].astype('category').cat.codes

df['Children']=df['Children'].astype('category').cat.codes
df['gender'].value_counts()

df['Married'].value_counts()

df['Children'].value_counts()

df['Channel1'].value_counts()

df['Channel1'].value_counts().count()
label_encoder = preprocessing.LabelEncoder() 

df['TVConnection']= label_encoder.fit_transform(df['TVConnection'])

df['Channel1']= label_encoder.fit_transform(df['Channel1'])

df['Channel2']= label_encoder.fit_transform(df['Channel2'])

df['Channel3']= label_encoder.fit_transform(df['Channel3'])

df['Channel4']= label_encoder.fit_transform(df['Channel4'])

df['Channel5']= label_encoder.fit_transform(df['Channel5'])

df['Channel6']= label_encoder.fit_transform(df['Channel6'])

df['Internet']= label_encoder.fit_transform(df['Internet'])

df['HighSpeed']= label_encoder.fit_transform(df['HighSpeed'])

df['AddedServices']= label_encoder.fit_transform(df['AddedServices'])

df['Subscription']= label_encoder.fit_transform(df['Subscription'])

df['PaymentMethod']= label_encoder.fit_transform(df['PaymentMethod'])





df['TotalCharges']=df['TotalCharges'].replace(r'^\s+$', 0, regex=True).astype(np.float64)
df.info()
dtest=pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')



dtest['gender']= label_encoder.fit_transform(dtest['gender'])

dtest['Married']= label_encoder.fit_transform(dtest['Married'])

dtest['Children']= label_encoder.fit_transform(dtest['Children'])

dtest['TVConnection']= label_encoder.fit_transform(dtest['TVConnection'])

dtest['Channel1']= label_encoder.fit_transform(dtest['Channel1'])

dtest['Channel2']= label_encoder.fit_transform(dtest['Channel2'])

dtest['Channel3']= label_encoder.fit_transform(dtest['Channel3'])

dtest['Channel4']= label_encoder.fit_transform(dtest['Channel4'])

dtest['Channel5']= label_encoder.fit_transform(dtest['Channel5'])

dtest['Channel6']= label_encoder.fit_transform(dtest['Channel6'])

dtest['Internet']= label_encoder.fit_transform(dtest['Internet'])

dtest['HighSpeed']= label_encoder.fit_transform(dtest['HighSpeed'])

dtest['AddedServices']= label_encoder.fit_transform(dtest['AddedServices'])

dtest['Subscription']= label_encoder.fit_transform(dtest['Subscription'])

dtest['PaymentMethod']= label_encoder.fit_transform(dtest['PaymentMethod'])





dtest['TotalCharges']=dtest['TotalCharges'].replace(r'^\s+$', 0, regex=True).astype(np.float64)
df.info()
df.head()
df.corr()
dtest.info()
df.corr()

df.info()



features=['SeniorCitizen','Married','Children','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','TVConnection','AddedServices','Subscription','tenure']

#scaler=RobustScaler()

scaler = MinMaxScaler()

#scaler=StandardScaler()



x=scaler.fit_transform(df[features])

y=scaler.fit_transform(df[['Satisfied']])



x_test=scaler.fit_transform(dtest[features])



from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(x,y,test_size=0.02,random_state=42) 
from sklearn.cluster import KMeans

km = KMeans(n_clusters=2)

km.fit(X_train,y_train)

y_pred=km.predict(X_val)





from sklearn.metrics import accuracy_score

print(accuracy_score(y_pred,y_val))

y_test=km.predict(x_test)
dtest['Satisfied'] =y_test

ans=dtest[['custId','Satisfied']].copy()

ans.to_csv('ans.csv',index=False)

ans.head(20)
ans.head(20)