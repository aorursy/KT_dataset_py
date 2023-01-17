import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/cpu-utilization/data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv')
df.head()
import plotly.express as px
fig = px.line(df, x='timestamp', y='value')
fig.show()
#Timestamps need to be marked as anamoly
#"realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv": [ "2014-02-19 00:22:00.000000", "2014-02-24 18:37:00.000000" ],
x = df['timestamp']
anomaly=[]
a=[]
i=0
for TS in x:
    if TS == '2014-02-19 00:22:00' or TS == '2014-02-24 18:37:00':  
     anomaly.insert(i,1)
     print("Record :",TS,"Index :", i)
     i=i+1
    else:
            anomaly.insert(i,0)
            i=i+1
            continue

df['target']=anomaly
df
#check the dataset whether values are placed correctly
df['target'].value_counts()
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
#Check the datatypes
df.dtypes
df.timestamp = pd.to_numeric(df.timestamp)
df.timestamp
#Testing and Training Dataset
#Ratio - 80:20
#Creating Testing and Training variables
from sklearn.model_selection import train_test_split
x=df.drop('target',axis=1)
y=df.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#Training 
x_train.shape
#Testing
x_test.shape
svc = SVC(kernel='rbf')
svc.fit(x_train, y_train)
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
y_pred = svc.predict(x_test)
print("Accuracy Score:",accuracy_score(y_test,y_pred))
print("\n")
print(classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
ds=pd.read_csv("../input/cpu-utilization/data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv")
ds.head
ds.timestamp = pd.to_numeric(df.timestamp)
ds.timestamp
from sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination = 1.0)
clf.fit(ds)
ds['anomaly']= pd.Series(clf.predict(ds))
ds['anomaly'].value_counts()
x = ds['anomaly']
anomaly=[]
a=[]
i=0
for TS in x:
    if TS == -1:  
     anomaly.insert(i,"No")
     i=i+1
    else:
            anomaly.insert(i,"Yes")
            i=i+1
            continue   

ds['anomaly']=anomaly
print(ds['anomaly'].value_counts())