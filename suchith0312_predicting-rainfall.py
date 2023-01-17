# importing the basic libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# IMPORTING other libraires which will be used
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
train=pd.read_csv("../input/train_2013.csv")
test=pd.read_csv("../input/test_2014.csv")

print("Training Size : (%d,%d)"%train.shape)
print("Test Size : (%d,%d)"%test.shape)
train.head(10)
train.columns
test.columns
train['TimeToEnd'][6]
train.info()
sample=pd.read_csv("sampleSubmission.csv")

sample.head(10)
sample.columns
plt.subplots(figsize=(20,20))
sns.distplot(train['Expected'].head(100))
ans=pd.DataFrame(columns=sample.columns)
ans['Id']=test['Id']
ans.head(10)
cols=list(sample.columns)
cols.remove('Id')
ans[cols]=1                  # making each probability as 1.
ans.head(10)
ans.to_csv("No_rain.csv",index=False)   # got an score of private:0.01025920 and public:0.01017651  
print("Done")
length=train.shape[0]
print(length)
# length of predicted0 and predicted1 and all these are same
for i in cols:
    l=len(i)
    if(l==10):
        k=int(i[l-1])
    else:                     # for handling two digits like 11...
        a=int(i[9])
        b=int(i[10])
        k=a*10+b
    ans[i]=(train.loc[train['Expected']<=k,'Expected'].value_counts()/(length)).sum()

print("Done")
ans.head(10)
ans.to_csv("only_train_per.csv",index=False)           
print("Done")
len(train.loc[train['Expected']>69])
# removing the examples which have greater than 69mm ranifall
train.drop((train.loc[train['Expected']>69]).index,inplace=True)
train.shape[0]
# converting the Expected values to classes.
train.loc[train['Expected']==0.0,'Expected']=0

for i in range(69):              # max value will go to 68
    train.loc[(train['Expected']>i) & (train['Expected']<=(i+1)),'Expected']=(i+1)
    
train['Expected']=train['Expected'].astype(int)
train.loc[(train['Expected']==68),'Expected']
plt.subplots(figsize=(15,9))
plt.xticks(rotation='90')                   # for rotation of 90 degree
sns.countplot(train['Expected'])
# k=list(map(float,train['RR1'][6].split()))
l=[]                                                    # empty list 
for i in train.index:
    k=list(map(float,train['RR1'][i].split()))
    k=[0 if (x==-99900.0 or x==-99901.0 or x==-99903.0) else x for x in k]
    mean=sum(k)/len(k)
    l.append(mean)
rr1=np.array(l)
rr1.shape=(train.shape[0],1)
print(rr1.shape)
plt.subplots(figsize=(15,9))
plt.scatter(rr1[0:500,:],train['Expected'].head(500),color='blue')
plt.xlabel("RR1")
plt.ylabel("Expected Rainfall")
plt.show()
l=[]
for i in test.index:
    k=list(map(float,test['RR1'][i].split()))
    k=[0 if (x==-99900.0 or x==-99901.0 or x==-99903.0) else x for x in k]
    mean=sum(k)/len(k)
    l.append(mean)
rr1_test=np.array(l)
rr1_test.shape=(test.shape[0],1)
print(rr1_test.shape)
# converting rr2 into mean values in train data
j=[]                                                    # empty list 
for i in train.index:
    k=list(map(float,train['RR2'][i].split()))
    k=[0 if (x==-99900.0 or x==-99901.0 or x==-99903.0) else x for x in k]
    mean=sum(k)/len(k)
    j.append(mean)
rr2=np.array(j)
rr2.shape=(train.shape[0],1)
print(rr2.shape)
plt.subplots(figsize=(15,9))
plt.scatter(rr2[0:500,:],train['Expected'].head(500),color='blue')
plt.xlabel("RR2")
plt.ylabel("Expected Rainfall")
plt.show()
# converting rr2 values into mean values in test data 
j=[]                                                    # empty list 
for i in test.index:
    k=list(map(float,test['RR2'][i].split()))
    k=[0 if (x==-99900.0 or x==-99901.0 or x==-99903.0) else x for x in k]
    mean=sum(k)/len(k)
    j.append(mean)
rr2_test=np.array(j)
rr2_test.shape=(test.shape[0],1)
print(rr2_test.shape)
j=[]                                                    # empty list 
for i in train.index:
    k=list(map(float,train['RR3'][i].split()))
    k=[0 if (x==-99900.0 or x==-99901.0 or x==-99903.0) else x for x in k]
    mean=sum(k)/len(k)
    j.append(mean)
rr3=np.array(j)
rr3.shape=(train.shape[0],1)
print(rr3.shape)
plt.subplots(figsize=(15,9))
plt.scatter(rr3[0:500,:],train['Expected'].head(500),color='blue')
plt.xlabel("RR3")
plt.ylabel("Expected Rainfall")
plt.show()
j=[]                                                    # empty list 
for i in test.index:
    k=list(map(float,test['RR2'][i].split()))
    k=[0 if (x==-99900.0 or x==-99901.0 or x==-99903.0) else x for x in k]
    mean=sum(k)/len(k)
    j.append(mean)
rr3_test=np.array(j)
rr3_test.shape=(test.shape[0],1)
print(rr3_test.shape)
p=[]
for i in train.index:
    k=list(map(float,train['RadarQualityIndex'][i].split()))
    k=[0.0 if x==999.0 else x for x in k]
    m=sum(k)/float(len(k))
    p.append(m)
RQi=np.array(p)
RQi.shape=(train.shape[0],1)
print(RQi.shape)
plt.subplots(figsize=(15,9))
plt.scatter(RQi[0:500,:],train['Expected'].head(500),color='blue')
plt.xlabel("Radar Quality Index")
plt.ylabel("Expected Rainfall")
plt.show()
p=[]
for i in test.index:
    k=list(map(float,test['RadarQualityIndex'][i].split()))
    k=[0.0 if x==999.0 else x for x in k]
    m=sum(k)/float(len(k))
    p.append(m)
RQi_test=np.array(p)
RQi_test.shape=(test.shape[0],1)
print(RQi.shape)
train['TimeToEnd'][0].count(" ")+1
numberofscans=[]
for i in train.index:
    number=train['TimeToEnd'][i].count(" ")+1
    numberofscans.append(number)

numberofscans=np.array(numberofscans).reshape(train.shape[0],1)
print(numberofscans.shape)
plt.subplots(figsize=(15,9))
plt.scatter(numberofscans,y.reshape(train.shape[0],1))
plt.xlabel("NUMBER OF RADAR SCANS")
plt.ylabel("RAINFALL")
numberofscans_test=[]
for i in test.index:
    number=test['TimeToEnd'][i].count(" ")+1
    numberofscans_test.append(number)

numberofscans_test=np.array(numberofscans_test).reshape(test.shape[0],1)
print(numberofscans_test.shape)
r=[]
for i in train.index:
    k=list(map(float,train['ReflectivityQC'][i].split()))
    k=[0 if (x==-99900.0 or x==-99901.0 or x==-99903.0) else x for x in k]
    m=sum(k)/len(k)
    r.append(m)

reflectivity=np.array(r).reshape(train.shape[0],1)
print(reflectivity.shape)
r=[]
for i in test.index:
    k=list(map(float,test['ReflectivityQC'][i].split()))
    k=[0 if (x==-99900.0 or x==-99901.0 or x==-99903.0) else x for x in k]
    m=sum(k)/len(k)
    r.append(m)
    
reflectivity_test=np.array(r).reshape(test.shape[0],1)
print(reflectivity_test.shape)
r=[]
for i in train.index:
    k=list(map(float,train['HybridScan'][i].split()))
    k=[0 if (x==-99900.0 or x==-99901.0 or x==-99903.0) else x for x in k]
    m=sum(k)/len(k)
    r.append(m)

hybrid=np.array(r).reshape(train.shape[0],1)
print(hybrid.shape)
r=[]
for i in test.index:
    k=list(map(float,test['HybridScan'][i].split()))
    k=[0 if (x==-99900.0 or x==-99901.0 or x==-99903.0) else x for x in k]
    m=sum(k)/len(k)
    r.append(m)
    
hybrid_test=np.array(r).reshape(test.shape[0],1)
print(hybrid_test.shape)
X=np.hstack((rr1,rr2,RQi,numberofscans,reflectivity,hybrid))
print(X.shape)
print(y.shape)
X_test=np.hstack((rr1_test,rr2_test,RQi_test,numberofscans_test,reflectivity_test,hybrid_test))
print(X_test.shape)
# xgb=XGBClassifier()
# xgb.fit(X,y)
# xgb_predict=xgb.predict_proba(X_test)
# print(xgb_predict.shape)
# temp=xgb_predict[:,68].reshape(test.shape[0],1)
# xgb_predict[:,68]=0.0
# xgb_predict=np.hstack((xgb_predict,temp))
# print(xgb_predict.shape)
# xgb_predict=np.cumsum(xgb_predict,axis=1)
# print(xgb_predict.shape)
# hybrid_data=pd.DataFrame(xgb_predict,columns=cols)
# hybrid_data.head(10)
# hybrid_data=pd.concat([test['Id'],hybrid_data],axis=1)
# hybrid_data.to_csv("hybrid.csv",index=False)
# print("Done")
# rf=RandomForestClassifier(n_estimators=10,random_state=42)
# rf.fit(X,y)
# f=rf.predict_proba(X_test)
# f.shape
# temp=f[:,68].reshape(test.shape[0],1)
# f[:,68]=0
# f=np.hstack((f,temp))
# f=np.cumsum(f,axis=1)
# print(f.shape)
# hybrid_rf=pd.DataFrame(f,columns=cols)
# hybrid_rf=pd.concat([test['Id'],hybrid_rf],axis=1)
# hybrid_rf.head(10)
# hybrid_rf[hybrid_rf[cols]>1]=1
# hybrid_rf.to_csv("hybrid_rf.csv",index=False)
# print(hybrid_data.shape,hybrid_rf.shape)
# ensemble=hybrid_data[cols]*0.8+hybrid_rf[cols]*0.2
# ensemble.head(10)
# ensemble=pd.concat([test['Id'],ensemble],axis=1)
# ensemble.to_csv("random_forest+Xgboost.csv",index=False)
# print("Done")