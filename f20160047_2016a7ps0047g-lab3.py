import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,auc,roc_auc_score,make_scorer
df= pd.read_csv('train.csv')

df.head()
df.dtypes
df.isnull().sum()
df=pd.get_dummies(df, prefix=None, prefix_sep='_', dummy_na=False, columns=['gender','SeniorCitizen','Married','Children','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed','AddedServices','Subscription','PaymentMethod'], sparse=False, drop_first=False, dtype=None)
df['TotalCharges']=df['TotalCharges'].apply(lambda x:x.split(' ')[0]).replace(to_replace=[''], value=0).apply(lambda x:float(x))
df.describe()
df.dtypes
df.columns
'''

features=['custId', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Satisfied',

       'gender_Female', 'gender_Male', 'SeniorCitizen_0', 'SeniorCitizen_1',

       'Married_No', 'Married_Yes', 'Children_No', 'Children_Yes',

       'TVConnection_Cable', 'TVConnection_DTH', 'TVConnection_No',

       'Channel1_No', 'Channel1_No tv connection', 'Channel1_Yes',

       'Channel2_No', 'Channel2_No tv connection', 'Channel2_Yes',

       'Channel3_No', 'Channel3_No tv connection', 'Channel3_Yes',

       'Channel4_No', 'Channel4_No tv connection', 'Channel4_Yes',

       'Channel5_No', 'Channel5_No tv connection', 'Channel5_Yes',

       'Channel6_No', 'Channel6_No tv connection', 'Channel6_Yes',

       'Internet_No', 'Internet_Yes', 'HighSpeed_No', 'HighSpeed_No internet',

       'HighSpeed_Yes', 'AddedServices_No', 'AddedServices_Yes',

       'Subscription_Annually', 'Subscription_Biannually',

       'Subscription_Monthly', 'PaymentMethod_Bank transfer',

       'PaymentMethod_Cash', 'PaymentMethod_Credit card',

       'PaymentMethod_Net Banking']

       '''

features=[

       'TVConnection_No',

       'Channel1_Yes',



       'Channel3_Yes',

       'Channel4_Yes',



       'Channel6_Yes',

       'Internet_Yes',

       'HighSpeed_Yes',

       'Subscription_Annually', 'Subscription_Biannually',

       'Subscription_Monthly', 'PaymentMethod_Bank transfer',

       'PaymentMethod_Cash', 'PaymentMethod_Credit card',

       'PaymentMethod_Net Banking']

X=df[features]

y=df['Satisfied']

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=69)

sns.set(color_codes=True)





sns.clustermap(X, method="ward", col_cluster=False, cbar_kws={'label': ''})

from sklearn.cluster import Birch

br=Birch(threshold=0.5, branching_factor=50, n_clusters=2, compute_labels=True, copy=True)

br.fit(X)
pre=br.predict(X_val)

err = roc_auc_score(y_val, pre)

print(err)
test_data=pd.read_csv('test.csv')

test_data.head()
test_data.isnull().sum()
test_data=pd.get_dummies(test_data, prefix=None, prefix_sep='_', dummy_na=False, columns=['gender','SeniorCitizen','Married','Children','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed','AddedServices','Subscription','PaymentMethod'], sparse=False, drop_first=False, dtype=None)
test_data['TotalCharges']=test_data['TotalCharges'].apply(lambda x:x.split(' ')[0]).replace(to_replace=[''], value=0).apply(lambda x:float(x))
test_data.head()
X_test=test_data[features]

#X_test=preprocessing.scale(X_test)

predicted=br.fit_predict(X_test)
predicted=predicted-1

predicted=predicted*-1
test_data['Satisfied']=np.array(predicted)

out=test_data[['custId','Satisfied']]

out=out.astype(int)

out.to_csv('submit.csv',index=False)