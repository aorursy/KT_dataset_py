import numpy as np
import pandas as pd
from sklearn import preprocessing
from datetime import datetime
import seaborn as sns
data = pd.read_csv("./352944080639365.csv")

corr = data.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
data.info()
data.head()

 #np.unique(data.iloc[:,11],return_counts=True)
# data.replace(['none'], False,inplace=True)
# data.replace(['wi-fi'],True,inplace=True)
# #for samsungsm-a910f
# timestamp = 1508308200

# #
# data.iloc[:,5] = data.iloc[:,5]-timestamp
dt  = datetime.fromtimestamp(data.iloc[1,5]/1000)
dt
data_discharge = data[data.iloc[:,14]==False]
data_charge = data[data.iloc[:,14]==True]
data_charge  = data_charge.iloc[:,[4,5,6,7,10,11,12]]
data_charge.head()
def data_clean(data):
    data.replace(['none'], False,inplace=True)
    data.replace(['wi-fi'],True,inplace=True)
    
    #data.iloc[:,4] = data.iloc[:,4]/1000
    #data.iloc[:,7] = data.iloc[:,7]/100
    #for samsungsm-a910f
    #timestamp = 1508308200000
    #2
    timestamp=1499668200000
    #3
    #timestamp = 1508913000000
    #4
    #timestamp = 1520663400000
    data.iloc[:,5] = data.iloc[:,5]-timestamp
    
    
    
    data_discharge = data[data.iloc[:,14]==False]
    data_charge = data[data.iloc[:,14]==True]
    
    data_charge = data_charge.iloc[:,[5,4,6,7,10,11,12]]
    data_discharge = data_discharge.iloc[:,[5,4,6,7,10,11,12]]
    
    return data_charge,data_discharge
x,y = data_clean(data)
y.head()
# y.iloc[:,0]=y.iloc[:,0]-min(y.iloc[:,0])
# x.iloc[:,0]=x.iloc[:,0]-min(x.iloc[:,0])
#y.iloc[:,1:]
ls = [0]
ls2 = [0]
dif = 0
charge = 0
for i in range(y.shape[0]-1):
    #print(y.iloc[i+1,0])
    charge+=(y.iloc[i+1,0]-y.iloc[i,0])*y.iloc[i,6]#*y.iloc[i,5]
    ls.append((y.iloc[i+1,0]-y.iloc[i,0]*y.iloc[i,6])/1000000)
    ls2.append(y.iloc[i,5]/y.iloc[i,6])
    if y.iloc[i,3]-y.iloc[i+1,3]==1:
           dif+=1
    if dif==100:
        charge_limit = charge
y['charge'] = ls
y['v/i'] = ls2
y.head()
y.shape
#charge_limit = 24875624182
charge_limit = charge_limit/3600000
charge_limit = charge_limit/2
print(charge_limit)

5000/3454.94
5000/2833.7337681944446
5000/4315.313589722222
3300/2230.8774704166667
4000/2767
import matplotlib.pyplot as plt
from keras.layers import Dense,Activation,Input
from keras.models import Sequential,Model
from keras import optimizers
# model = Sequential()
# model.add(Dense(16, input_dim=(6)))
# model.add(Activation('relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(1,activation='relu'))

# model.summary()


inp = Input(shape=(8,))
h1 = Dense(32,activation='relu')(inp)
h2 = Dense(32,activation='relu')(h1)
h3 = Dense(32,activation='relu')(h2)
h4 = Dense(32,activation='relu')(h3)
h5 = Dense(1)(h4)

out = Activation('relu')(h5)

model = Model(inputs=inp,outputs=out)

model.summary()
adam = optimizers.adam(lr = 0.1)
model.compile(loss='mae', optimizer='adam' ,metrics=['accuracy'])
hist = model.fit(y.iloc[:,1:],y.iloc[:,0],epochs=100,shuffle=True,batch_size=256,validation_split=0.20)

plt.figure(1)
plt.plot(hist.history['acc'], color = 'r')
plt.plot(hist.history['val_acc'], color = 'b')

plt.figure(2)
plt.plot(hist.history['loss'], color = 'r')
plt.plot(hist.history['val_loss'], color = 'b')


model.predict(y.iloc[:,1:])

import numpy as np
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(y.iloc[:,1:],y.iloc[:,0])
reg.score(y.iloc[:,1:],y.iloc[:,0])
reg.coef_
reg.score(y.iloc[:,1:],y.iloc[:,0])
ans = reg.predict(y.iloc[:,1:])
np.mean(ans)

np.mean(y.iloc[:,0])
np.asarray(y.iloc[1,1:])
[3000,False,15,12,27.9,3903,990]
td = datetime.fromtimestamp(1570599938)
print(int(td.strftime('%s')))
import seaborn as sns
sns.set()
ax = sns.heatmap(y.iloc[:,4:])
1499668200000-71367547723.42853





.78
.475584130932088
-2.3
.53

.67
.63
-.83
.38

.21
-.18
.27
.15


data = pd.read_csv(".csv")

date = np.mean(data.iloc[:,5])
date
mn = np.mean(data.iloc[:,5])
mn
x,y = data_clean(data)

#y.iloc[:,1:]
ls = [0]
ls2 = [0]
dif = 0
charge = 0
for i in range(y.shape[0]-1):
    #print(y.iloc[i+1,0])
    charge+=((y.iloc[i,0]-y.iloc[i+1,0])*y.iloc[i,6])
    ls.append(((y.iloc[i,0]-y.iloc[i+1,0])*y.iloc[i,6]))
    if y.iloc[i,6]!=0:
        ls2.append(y.iloc[i,5]/y.iloc[i,6])
    if y.iloc[i,6]==0:
        ls2.append(0)
    if y.iloc[i,3]-y.iloc[i+1,3]==1:
           dif+=1
    if dif==100:
        charge_limit = charge

print(np.asarray(ls).shape) 
print(y.shape)
y['charge'] = ls
y['v/i'] = ls2

y.head()
reg.score(y.iloc[:,1:],y.iloc[:,0])
y.head()
ans = reg.predict(y.iloc[:,1:])
aa = np.mean(ans)
aa

a = np.mean(y.iloc[:,0])
a
ansfinal = date-aa
int(ansfinal)
td = datetime.fromtimestamp(int(ansfinal)/1000)
td



