import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing.data 
train_data 
test_data 
pp = train_data[:,:13]
pp_1 = pd.DataFrame(pp)


preprocessing.__dict__
scalar = StandardScaler()
scalar.fit(pp)
pp = scalar.transform(pp)

pp_1 = pd.DataFrame(pp)

y = train_data[:,-1]
a = []
for i in range(379):
    a.append(1)
pp_1.insert(13,'c',a)
pp_1.insert(14,'y',y)
train = pp_1.to_numpy()

traindf = pd.DataFrame(train)

x = train[0,:14]
x
def step_grad(points,learn_rate,m):
    m = np.array(m)
    m_slope = [0 for i in range(14)]
    
    M = len(points)
    a = 0
    for i in range(M):
        
        for j in range(14):
            x = points[i,:14]
            y = points[i,-1]
            
            x1 = points[i,j]
            m_slope[j] += (-2/M)*(y-(m*x).sum())*x1
            
    m_slope = np.array(m_slope)
    new_m = m - learn_rate*m_slope
    new_m_list = list(new_m)
    return new_m_list
    
def gd(points,learn_rate,num_iter):
    m = []
    for i in range(14):
        m.append(0)
    for i in range(num_iter):
        if i < 200:
            m = step_grad(points,learn_rate,m)
        if i>200 and i<=306:
            m = step_grad(points,0.17,m)
        
        
        if i>306:
            m = step_grad(points,0.1,m)
        print(i,'cost:',cost(points,m))   
    return m

        
    
def cost(points,m):
    m = np.array(m)
    total_cost = 0
    M = len(points)
    for i in range(M):
        y = points[i,-1]
        x = points[i,:14]
        x = np.array(x)
        mx = m*x
        sum_mx = mx.sum()
        total_cost += (1/M)*((y-sum_mx)**2)
    return total_cost
def run():
    
    
    learn_rate = 0.159
    learn_rate2 = 0.05
    num_iter = 400
    m = gd(train,learn_rate,num_iter)
    return m
train_m = run()

c = train_m[-1]
j = 0
train_m1 = train_m[:13]
y_train = []
for i in range(len(train)):
    j +=1
    x = train[i,:13]
    
    x = np.array(x)
    train_m1 = np.array(train_m1)
    p= train_m1*x
    sum_p = p.sum()
    final = sum_p+c
    y_train.append(final)

        
y_train
import matplotlib.pyplot as plt
y = train_data[:,-1]
y = list(y)
plt.scatter(y_train,y)
plt.axis([0,50,0,50])
plt.show()
y = np.array(y)
y_train = np.array(y_train)
a = (y_train)
plt.show()

y_train = list(y_train)
y_train

df = pd.DataFrame(test_data)
a = []
pp = test_data[:,:13]
pp_1 = pd.DataFrame(pp)
#scalar = StandardScaler()
#scalar.fit(pp)
pp = scalar.transform(pp)
pp_1 = pd.DataFrame(pp)
for i in range(127):
    a.append(1)
pp_1.insert(13,'c',a)

test = pp_1.to_numpy()
traindf = pd.DataFrame(test)

train_m = run()

train_m
x = test[:,:13]
df = pd.DataFrame(x)
df
c = train_m[-1]
j = 0
train_m1 = train_m[:13]
y_train = []
for i in range(len(test)):
    j +=1
    x = test[i,:13]
    
    x = np.array(x)
    train_m1 = np.array(train_m1)
    p= train_m1*x
    sum_p = p.sum()
    final = sum_p+c
    y_train.append(final)