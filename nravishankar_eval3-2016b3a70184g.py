# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.cluster import KMeans, AgglomerativeClustering, Birch

from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# from sklearn.preprocessing import LabelEncoder 

  

# le = LabelEncoder() 

  

# df['gender']= le.fit_transform(df['gender']) 

# df['Married']= le.fit_transform(df['Married']) 

# df['Children']= le.fit_transform(df['Children']) 

# df['Internet']= le.fit_transform(df['Internet']) 

# df['AddedServices']= le.fit_transform(df['AddedServices']) 

# df['HighSpeed']= le.fit_transform(df['HighSpeed']) 



# df['Channel1'] = df['Channel1'].str.replace("No tv connection", "No")

# df['Channel2'] =df['Channel2'].str.replace("No tv connection", "No")

# df['Channel3'] =df['Channel3'].str.replace("No tv connection", "No")

# df['Channel4'] =df['Channel4'].str.replace("No tv connection", "No")

# df['Channel5'] =df['Channel5'].str.replace("No tv connection", "No")

# df['Channel6'] =df['Channel6'].str.replace("No tv connection", "No")



# test = pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')

# test['Channel1'] = test['Channel1'].str.replace("No tv connection", "No")

# test['Channel2'] =test['Channel2'].str.replace("No tv connection", "No")

# test['Channel3'] =test['Channel3'].str.replace("No tv connection", "No")

# test['Channel4'] =test['Channel4'].str.replace("No tv connection", "No")

# test['Channel5'] =test['Channel5'].str.replace("No tv connection", "No")

# test['Channel6'] =test['Channel6'].str.replace("No tv connection", "No")





# test['gender']= le.fit_transform(test['gender']) 

# test['Married']= le.fit_transform(test['Married']) 

# test['Children']= le.fit_transform(test['Children']) 

# test['Internet']= le.fit_transform(test['Internet']) 

# test['AddedServices']= le.fit_transform(test['AddedServices']) 

# test['HighSpeed']= le.fit_transform(test['HighSpeed']) 



# test['Channel1']= le.fit_transform(test['Channel1']) 

# test['Channel2']= le.fit_transform(test['Channel2']) 

# test['Channel3']= le.fit_transform(test['Channel3']) 

# test['Channel4']= le.fit_transform(test['Channel4']) 

# test['Channel5']= le.fit_transform(test['Channel5']) 

# test['Channel6']= le.fit_transform(test['Channel6']) 



# p = pd.get_dummies(test['TVConnection'])

# test['Cable'] = p['Cable']

# test['DTH'] = p['DTH']

# test['No_TV'] = p['No']



# p2 = pd.get_dummies(test['Subscription'])

# test['Monthly'] = p2['Monthly']

# test['Annually'] = p2['Annually']

# test['Biannually'] = p2['Biannually']



# p3 = pd.get_dummies(test['PaymentMethod'])

# test['Cash'] = p3['Cash']

# test['Bank_Transfer'] = p3['Bank transfer']

# test['Net_Banking'] = p3['Net Banking']

# test['Credit_Card'] = p3['Credit card']



# test = test.drop('TVConnection', axis =1)

# test = test.drop('Subscription', axis =1)

# test = test.drop('PaymentMethod', axis = 1)



# test.loc[71, 'TotalCharges'] = 2298.24

# test.loc[580, 'TotalCharges'] = 2298.24

# test.loc[637, 'TotalCharges'] = 2298.24

# test.loc[790, 'TotalCharges'] = 2298.24

# test.loc[1505, 'TotalCharges'] = 2298.24



# test['TotalCharges'] = test['TotalCharges'].astype(float)

# test = test.drop('custId', axis = 1)

# test

# kmeans = KMeans(n_clusters=2)

# pred_y = kmeans.fit_predict(test)

# from sklearn.cluster import AgglomerativeClustering

# ag =  AgglomerativeClustering(affinity = 'euclidean', linkage = 'ward', n_clusters=2)

# pred3_y = ag.fit_predict(test)

# kmeans = KMeans(n_clusters=10).fit(test)

# pred4_y = [y[kmeans.predict(test)==i].value_counts().sort_values(ascending=False).index[0] for i in range(10)]
# chmap={'Yes':1, 'No':0, 'No tv connection':0}

# df['Channel1']=df['Channel1'].map(chmap)

# df['Channel2']=df['Channel2'].map(chmap)

# df['Channel3']=df['Channel3'].map(chmap)

# df['Channel4']=df['Channel4'].map(chmap)

# df['Channel5']=df['Channel5'].map(chmap)

# df['Channel6']=df['Channel6'].map(chmap)



# dftest['Channel1']=dftest['Channel1'].map(chmap)

# dftest['Channel2']=dftest['Channel2'].map(chmap)

# dftest['Channel3']=dftest['Channel3'].map(chmap)

# dftest['Channel4']=dftest['Channel4'].map(chmap)

# dftest['Channel5']=dftest['Channel5'].map(chmap)

# dftest['Channel6']=dftest['Channel6'].map(chmap)

# df['channelnew']=df['Channel3']+df['Channel4']+df['Channel5']+df['Channel6']

# df=df.drop(['Channel1', 'Channel2', 'Channel3','Channel4','Channel5', 'Channel6'], axis=1)



# dftest['channelnew']=dftest['Channel3']+dftest['Channel4']+dftest['Channel5']+dftest['Channel6']

# dftest=dftest.drop(['Channel1', 'Channel2', 'Channel3','Channel4','Channel5', 'Channel6'], axis=1)



# hmap={'Monthly':1, 'Biannually':6, 'Annually':12}

# df['Subscription']=df['Subscription'].map(hmap)



# dftest['Subscription']=dftest['Subscription'].map(hmap)
df = pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

df2 = pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

df.loc[df['TotalCharges'] == " ", "TotalCharges"] = df["MonthlyCharges"]

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

df.head()
df_test = pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")

df_test.loc[df_test['TotalCharges'] == " ", "TotalCharges"] = df_test["MonthlyCharges"]

df_test['TotalCharges'] = pd.to_numeric(df_test['TotalCharges'])



dft = df_test.drop('custId', axis = 1)

dft = pd.get_dummies(dft)



dft.head()

y = df['Satisfied']

df = df.drop(['Satisfied'], axis = 1)

df = df.drop(['custId'], axis = 1)



df = pd.get_dummies(df)

columns = [ 'MonthlyCharges', 'TotalCharges','Married_Yes', 

       'Children_No', 'TVConnection_Cable','Channel1_No tv connection','Channel1_No',

       'Channel2_No tv connection', 'Channel3_No', 'Channel3_No tv connection',

        'Channel4_No','Channel4_No tv connection', 'Channel5_No',

       'Channel5_No tv connection', 'Channel6_No',

       'Channel6_No tv connection', 'AddedServices_No', 

       'Subscription_Annually', 'Subscription_Biannually', 'Subscription_Monthly']



df = df[columns]

dft = dft[columns]

dft = StandardScaler().fit_transform(dft)

df = StandardScaler().fit_transform(df)

bir = Birch(n_clusters=2).fit(df)

pred = bir.predict(dft)

where_0 = np.where(pred == 0)

where_1 = np.where(pred == 1)



pred[where_0] = 1

pred[where_1] = 0

result = pd.DataFrame({'Satisfied':pred}, index=df_test['custId'])

result.to_csv("sub5.csv")