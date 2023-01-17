# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib as mpl
import matplotlib.pyplot as plt 
from  matplotlib.animation import FuncAnimation

from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.preprocessing import MinMaxScaler

boston=load_boston()
print(boston.DESCR)
features=pd.DataFrame(boston.data,columns=boston.feature_names)
features
target=pd.DataFrame(boston.target,columns=['target'])
target
max(target['target'])
min(target['target'])
df=pd.concat([features,target],axis=1)
df
df.describe().round(decimals=2)
corr=df.corr('pearson')
corrs=[abs(corr[attr]['target']) for attr in list(features)]
l= list(zip(corrs,list(features)))

l.sort(key=lambda x : x[0], reverse=True)
corrs,labels=list(zip((*l)))
index=np.arange(len(labels))

plt.figure(figsize=(15, 5))
plt.bar(index,corrs,width=0.5)
plt.xlabel('Attributes')
plt.ylabel('correlation with target')
plt.xticks(index,labels)
plt.show()
X=df['LSTAT'].values
Y=df['target'].values
print (Y[:5])
x_scaler=MinMaxScaler()
X=x_scaler.fit_transform(X.reshape(-1,1))
X=X[:,-1]
y_scaler=MinMaxScaler()
Y= y_scaler.fit_transform(Y.reshape(-1,1))
Y=Y[:,-1]
print(Y[:5])
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2)
def error(m,x,c,t):
    N=x.size
    e=sum(((m*x+c)-t))
    return e*1/(2*N)
def update (m,x,c,t,learning_rate):
    grad_m=sum(2*((m*x+c)-t)*x)
    grad_c=sum(2*((m*x+c)-t))
    m=m-grad_m*learning_rate 
    c=c-grad_c*learning_rate 
    return m,c
def gradient_descent(init_m,init_c,x,t,learning_rate,iterations,error_threshold):
    m=init_m
    c=init_c
    error_value=list()
    values_mc=list()
    for i in range(iterations):
        e=error(m,x,c,t)
        if e<error_threshold:
            print ("error less than threshold.stopping gradient descent")
            break
        error_value.append(e)
        m,c=update(m,x,c,t,learning_rate)
        values_mc.append(m)
    return m,c,error_value,values_mc
    
#%time
init_m=0.9
init_c=0
learning_rate=0.001
iterations=250
error_threshold=0.001
m,c,error_value,values_mc=gradient_descent(init_m,init_c,xtrain,ytrain,learning_rate,iterations,error_threshold)
plt.scatter(xtrain,ytrain,color='b')
plt.plot(xtrain,m*xtrain+c,color='r')
plt.plot(np.arange(len(error_value)),error_value)
plt.xlabel('error')
plt.ylabel('iterations')
mc_values_anim= values_mc[0:250:5]

fig , ax=plt.subplots()
ln,=plt.plot([],[],'ro-',animated=True)
def init():
    plt.scatter(xtest,ytest,color='g')
    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.0)
    return ln,
def update_frame(frame):
    m,c= mc_values_anim[frame]
    x1,y1=-0.5,m*(-0.5)+c
    x2,y2=1.5,m*1.5+c
    ln.set_data([x1,x2],[y1,y2])
    return ln,
    
anim=FuncAnimation(fig,update_frame,frames=range(len(mc_values_anim)),init_func=init,blit=True)
    
predicted=(m*xtest+c)
mean_squared_error(ytest,predicted)
p=pd.DataFrame(list(zip(xtest,ytest,predicted)),columns=['x','target_x','pridected_y'])
p.head()
plt.scatter(xtest,ytest,color='b')
plt.plot(xtest,predicted,color='r')