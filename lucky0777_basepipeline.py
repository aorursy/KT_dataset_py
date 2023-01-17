import numpy as np
import matplotlib.pyplot as plt
path='/kaggle/input/bid-data/'
bid=np.loadtxt(path+'data.csv',np.int32)
bid=np.cumsum(bid,dtype=np.int32)
plt.plot(bid[:100])
print(len(bid))
def RN(data,step=1): # range chart
  n=0
  up=data[0]
  dn=up-2*step
  size=int(sum(abs(np.diff(data)))/step)+10
  tmp=np.zeros(size,np.int32)
  for i in range(len(data)):
    while data[i]>=up:
      tmp[n]=up
      up+=step
      dn+=step
      n+=1
    while data[i]<=dn:
      tmp[n]=dn
      dn-=step
      up-=step
      n+=1
  return tmp[:n]
rn=RN(bid,5)
plt.plot(rn[:100])
print(len(rn))
bid=rn
def MA(data,per=20): # moving average
  tmp=np.zeros(len(data),np.float64)
  tmp[per-1:]=np.convolve(data,[1/per]*per,'valid')
  return tmp
# ma=MA(rn)
# plt.plot(rn[:100])
# plt.plot(ma[:100])
# print(len(ma))
def DS(data,step=10): # downsampling
  n=0
  tmp=np.zeros(int(len(data)/step+1),data.dtype)
  for i in range(0,len(data),step):
    tmp[n]=data[i]
    n+=1
  return tmp[:n]
# ds=DS(ma)
# plt.plot(ds[:100])
# print(len(ds))
bid2=np.full_like(bid,1e9)
# imx=[]
# imn=[]
rn=19 # min range=rn+1
ud=0
mx=bid[0]
mn=bid[0]
for i in range(len(bid)):
  if bid[i]<mn:
    if ud>=0: bid2[ud]=1 #mx #imx.append(ud)
    ud=-i
    mn=bid[i]
    mx=mn+rn
  if bid[i]>mx:
    if ud<=0: bid2[-ud]=-1 #mn #imn.append(-ud)
    ud=i
    mx=bid[i]
    mn=mx-rn
bid2[-1]=0 #bid[-1]
del rn,ud,mx,mn
# imx=np.array(imx,np.int32)
# imn=np.array(imn,np.int32)
# zz=np.concatenate((imx,imn))
# zz=np.sort(zz)
z=0.
for i in range(len(bid)-1,-1,-1):
  if bid2[i]!=1e9: z=bid2[i]
  else: bid2[i]=z
del z
bid=np.diff(bid)
bid2=bid2[1:]
win=20 # input size, history depth, lags number
total=len(bid)-win+1
# input prepare
In = np.zeros((total,win),np.float32)
for i in range(total-1,-1,-1):
  In[i]=bid[i:i+win]
del bid
# output prepare
Out=np.copy(bid2[win:]) #np.copy(In[1:,-1])
In=In[:-1,:]# delete last row
# shuffle data
# np.random.seed(1)
# np.random.shuffle(In)
# np.random.seed(1)
# np.random.shuffle(Out)
# train validation split
sizeTr=int(total/2)
sizeVal=sizeTr
InTrain=In[:sizeTr]
InVal=-In[sizeTr:sizeTr+sizeVal]
OutTrain=Out[:sizeTr]
OutVal=-Out[sizeTr:sizeTr+sizeVal]
del In,Out
print(InTrain.shape,InVal.shape)
print(len(OutTrain),len(OutVal))
import gc
gc.collect()
import time
t = time.monotonic()
# random forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=-1, n_estimators=40, max_samples=2500, max_features=0.5, min_samples_leaf=5, oob_score=True)
model.fit(InTrain, OutTrain)
t=time.monotonic()-t
print(int(t/60),"m",int(t)%60,"s")
plt.plot(model.feature_importances_);
def rmse(x,y): return (((x-y)**2).mean())**0.5
def acc(x,y):
  tmp=x*y
  n=0
  for _,i in enumerate(tmp):
    if i>0: n+=1
  return n/len(tmp)
train=model.predict(InTrain)
val=model.predict(InVal)
print("train err:",rmse(train,OutTrain),"acc:",acc(train,OutTrain))
print("valid err:",rmse(val,OutVal),"acc:",acc(val,OutVal))
v0=InTrain[1:,-1]
v1=OutTrain[:-1]
v=np.cumsum(v0*v1)
plt.plot(v);
v[-1]/10798
plt.figure(figsize=(50,5))
plt.plot(OutTrain[:500])
plt.plot(train[:500]);
# v0=InTrain[1:,-1]
# print(p/len(v),n/len(v))
# print("EP",v[-1]/len(v))
# print("pos win",sp)
# print("win",s)
# print("loss",v[-1]-s)
# print("profit",v[-1])
# print("deals",len(v))
# plt.figure(figsize=(50,5))
# plt.plot(v[:],'b');