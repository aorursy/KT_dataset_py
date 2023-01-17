import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import math
from scipy import stats
df = pd.read_csv('../input/train.csv')
df = df[df['LotArea']<30000] # remove outliers
df.head()
x = df['LotArea']
max_x = x.max()
mean_x = x.mean()
x_norm = list(map(lambda elem: (elem-mean_x)/max_x,list(x)))
y = df['SalePrice']
max_y = y.max()
mean_y = y.mean()
y_norm = list(map(lambda elem: (elem-mean_y)/max_y,list(y)))

plt.figure(figsize=(10,10))
plt.scatter(x_norm,y_norm, alpha=0.6)
plt.show()
n = len(x)
a = 0.01 # [a,b] also known as theta
b = 1
alpha = 0.5
max_iter = 3000

def err (a,b,x,y): # J(theta) = 1/2n * (y - theta dot x + theta_0)^2
    est_y = list(map(lambda elem: a*elem+b,x)) # theta dot x + theta_0
    err = np.subtract(np.array(est_y),np.array(y)) # y^ - y
    err_2 = np.power(err,2)
    return (1/(2*n)) * sum(err_2)

def der(a,b,x,y): # gradient of J(theta) = [1/n * (y - theta dot x + theta_0) dot x, 1/n*(y - theta dot x + theta_0) dot 1]
    est_y = list(map(lambda elem: a*elem+b,x)) # theta dot x + theta_0
    err = np.subtract(np.array(est_y),np.array(y)) # y^ - y
    return (1/n * np.dot(err,x),1/n*sum(err))

err_iter = []
for i in range(0,max_iter): 
    deriv = der(a,b,x_norm,y_norm)
    a -= alpha*deriv[0]
    b -= alpha*deriv[1]
    err_iter.append(math.sqrt(err(a,b,x_norm,y_norm)))
    
index = list(range(0,max_iter))
#plt.figure(figsize=(10,10))
plt.plot(index,err_iter)
plt.show()

def y_from_x(x): # theta dot x + theta_0
    return a*x+b
plt.figure(figsize=(10,10))

plt.scatter(x_norm,y_norm, alpha=0.6)
plt.plot([0,1],[y_from_x(0),y_from_x(1)])
plt.show()
# R^2 is ESS / TSS, Explained sum of squares over total sum of squares
y_avg=sum(y_norm)/len(y_norm)
y_hat = np.apply_along_axis(y_from_x, 0, x_norm)
ESS = np.sum(np.add(y_hat, -1*y_avg)**2)
TSS = np.sum(np.add(y_norm, -1*y_avg)**2)

degf = len(y_norm)

se = math.sqrt(np.sum(np.add(y_norm,-y_hat)**2)/degf) / math.sqrt(np.sum(np.add(x_norm,-sum(x_norm)/len(x_norm))**2))

slope, intercept, r_value, p_value, std_err = stats.linregress(x_norm,y_norm)

tscore = (a - 0)/(se)
p = stats.t.sf(tscore,df=degf)
prova = stats.t.ppf(p_value,df=degf)
print("R Squared:", ESS/TSS)
print("p-value", p)
df_2 = pd.read_csv('../input/test.csv')
x_test = df_2['LotArea']
y_test = list(map(y_from_x,x))
output = pd.DataFrame(df_2['Id'])
output['SalePrice'] = pd.Series(y_test)
output.to_csv("output.csv", index=False)