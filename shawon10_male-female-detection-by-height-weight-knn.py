import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('../input/weight-height.csv')
data.head(10)
df=data
columns=['Gender']
df = df.drop(columns, axis=1)
df.head(10)
Xm=df[0:25]
Xf=df[5001:5026]
frame=[Xm,Xf]
X_train=pd.concat(frame)
X_train=np.array(X_train)
print(X_train)
X_train.shape
lbl=data['Gender']
ym=lbl[0:25]
yf=lbl[5001:5026]
frames=[ym,yf]
Y_train=pd.concat(frames)
Y_train=np.array(Y_train)
print(Y_train)
Y_train.shape
label=data['Gender']
lb=label[400:410]
lc=label[9000:9010]
frames=[lb,lc]
Y=pd.concat(frames)
Y=np.array(Y)
print(Y)
x = data['Height']
y=data['Weight']
male=x[0:25]
female=x[5001:5026]
wm=y[0:25]
wf=y[5001:5026]
plt.scatter(male,wm, color='b')
plt.scatter(female,wf,color='g')
plt.xlabel('Male')
plt.ylabel('Female')
plt.show()
testx=df[400:410]
testy=df[9000:9010]
frames=[testx,testy]
X=pd.concat(frames)
X=np.array(X)
print(X)
X.shape
def predict(input_feature_set, k):
    distances = []
    z=0
    for training_feature_set in X_train:
        group=Y_train[z]
        #print("Group=",group)
        #print("Training Feature=",training_feature_set)
        euclidean_distance = np.linalg.norm(np.array(input_feature_set) - np.array(training_feature_set))
        #print("Distance=",euclidean_distance)
        distances.append([euclidean_distance, group])
        z=z+1
        #print(z)

    nearest = sorted(distances)[:k]
    #print("Sorted=",nearest)
    votes=[]
    #votes = [d[1] for d in nearest] 
    for d in nearest:
        votes.append(d[1])
    #print(votes)
    #prediction = Counter(votes).most_common(1)[0][0]
    item={}
    for i in votes:
        if i in item:
            item[i]=item[i]+1
        else:
            item[i]=1
            
    #finding most common class        
    m=0
    for k in item:
        if item[k]>m:
            m=item[k]
    
    for k in item:
        if item[k]==m:
            index=k
            
    prediction=index
        

    return prediction
#Training
toutput=[]
for j in X_train:
    predicted=predict(j,3)
    toutput.append(predicted)
print("Output=",toutput)
print(len(toutput))
c=0
for p in range(0,50):
        if toutput[p]==Y_train[p]:
            c=c+1
result=(c/50.0)*100
print("Training Accuracy=",result,"%")
output=[]
for j in X:
    predicted=predict(j,3)
    output.append(predicted)
print("Output=",output)
c=0
for p in range(0,20):
        if output[p]==Y[p]:
            c=c+1
result=(c/20.0)*100
print("Testing Accuracy=",result,"%")
