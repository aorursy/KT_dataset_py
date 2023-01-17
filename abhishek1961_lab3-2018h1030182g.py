import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
df=pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

df.drop(['custId','PaymentMethod','Married','HighSpeed','Children','gender','AddedServices','Subscription','SeniorCitizen','tenure'],1,inplace=True)

df=pd.get_dummies(df,columns=['TVConnection','Channel1','Channel2','Channel3','Channel4',

                          'Channel5','Channel6','Internet' ],drop_first=True)
df.columns
df.info()
import os,sys

from scipy import stats

import numpy as np



f=open("/kaggle/input/eval-lab-3-f464/train.csv", 'r').readlines()



N=len(f)-1

for i in range(0,N):

    w=f[i].split()

    l1=w[1:8]

    l2=w[8:15]

    try:

        list1=[float(x) for x in l1]

        list2=[float(x) for x in l2]

    except (ValueError)as e:

        print ("error",e,"on line",i)

    result=stats.ttest_ind(list1,list2)

    print (result[1])

df.TotalCharges = df.TotalCharges.str.replace(' ','')

df.TotalCharges = pd.to_numeric(df.TotalCharges).fillna(0.0)
# X = StandardScaler().fit_transform(X)

# X = np.array(df.drop(['Satisfied','PaymentMethod','custId','gender'], 1).astype(float))

X = np.array(df.drop(['Satisfied'], 1).astype(float))

y = np.array(df['Satisfied'])
from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()

scaler2 = StandardScaler()

X_scaled = scaler2.fit_transform(X)
 
 

high_corr=df.corr()

high_corr['Satisfied']
 
kmeans = KMeans(n_clusters=2,max_iter=1000,random_state=42) 

kmeans.fit(X_scaled)
# from sklearn.cluster import Birch

# brc = Birch(branching_factor=1000, n_clusters=2, threshold=0.5,compute_labels=True)

# brc.fit(X_scaled) 
from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch

model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='single')

model.fit(X_scaled)
# correct = 0

# for i in range(len(X_scaled)):

#     predict_me = np.array(X[i].astype(float))

#     predict_me = predict_me.reshape(-1, len(predict_me))

#     prediction = kmeans.predict(predict_me)

# #     prediction = dbscan_predict(db,X_scaled)

    

#     if prediction[0] == y[i]:

#         correct += 1



# print(correct/len(X_scaled))
df2=pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")

custId=df2['custId']

# df2.drop(['TotalCharges','PaymentMethod','custId','gender'],1,inplace=True)

df2.drop([ 'custId','HighSpeed','Married','PaymentMethod','Children','gender','AddedServices','Subscription','SeniorCitizen','tenure'],1,inplace=True)

df2.TotalCharges = df2.TotalCharges.str.replace(' ','')

df2.TotalCharges = pd.to_numeric(df.TotalCharges).fillna(0.0)
df2=pd.get_dummies(df2,columns=['TVConnection','Channel1','Channel2','Channel3','Channel4',

                          'Channel5','Channel6','Internet'],drop_first=True)
 


# scaler = StandardScaler()

X_scaled2 = scaler2.fit_transform(df2)



X_scaled2
 
 
Y_dfpred=pd.DataFrame()

Y_dfpred['custId']=custId



pred=model.fit_predict(X_scaled2)



Y_dfpred['Satisfied']=list(pred)
Y_dfpred.to_csv('Output_8.csv',index=False) 
Y_dfpred
 
 