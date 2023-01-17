import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
data = np.random.randint(1,10,20)
sum(data)/len(data)
data.mean()
from statistics import median,mean,mode
data = np.random.randint(1,10,20)
data
mean(data)
median(data)
# data
mode([1,2,3,4,2,3,3,5])
plt.hist(data,rwidth=.95,align="right")
data2 = np.random.randint(1,20,50)
print("Mean is::",mean(data2))
print("Median is::",median(data2))
plt.figure(figsize=(16,4))
plt.subplot(1,2,1)
sns.distplot(data2,hist_kws={"rwidth":.98,"align":"left"})
plt.subplot(1,2,2)
sns.boxplot(data2,orient="v",width=.2)
plt.show()
data3 = np.append(data2,100)
mean(data3)
median(data3)
plt.figure(figsize=(16,4))
plt.subplot(1,2,1)
sns.distplot(data3,hist_kws={"rwidth":.98,"align":"left"})
plt.subplot(1,2,2)
sns.boxplot(data3,orient="v",width=.2)
plt.show()
data4 =  [40, 38, 42, 40, 39, 39, 43, 40, 39, 40]
data5 = [46, 37, 40, 33, 42, 36, 40, 47, 34, 45]
rang1 = max(data4)-min(data4)
rang2 = max(data5)-min(data5)
print(rang1)
print(rang2)
data6 = [46, 37, 40, 33, 42, 36, 40, 47, 34, 45]
data6_np = np.array(data6)
data6_np.mean()
data6_np-data6_np.mean()
(data6_np-data6_np.mean())**2
((data6_np-data6_np.mean())**2).sum()
(((data6_np-data6_np.mean())**2).sum())/(len(data6)-1)
np.sqrt((((data6_np-data6_np.mean())**2).sum())/(len(data6)-1))
#### For small size data
plt.figure(figsize=(16,8))
tmp = np.random.normal(loc=0,scale=1,size=3)
plt.subplot(2,2,1)
sns.distplot(tmp,color="m")
plt.title("small data size")

#### For midium size data
tmp = np.random.normal(loc=0,scale=1,size=40)
plt.subplot(2,2,2)
sns.distplot(tmp,color="b")
plt.title("medium data size")

#### For large size data
tmp = np.random.normal(loc=0,scale=1,size=100)
plt.subplot(2,2,3)
sns.distplot(tmp,color="c")
plt.title("large data size")

#### For Very large size data
tmp = np.random.normal(loc=0,scale=1,size=5000)
plt.subplot(2,2,4)
sns.distplot(tmp,color="r")
plt.title("very large data size")
plt.show()
#### For small size data
plt.figure(figsize=(16,8))
tmp = np.random.normal(loc=0,scale=1,size=3)
plt.subplot(2,2,1)
sns.distplot(tmp,color="m",label="small",hist=False)

#### For midium size data
tmp = np.random.normal(loc=0,scale=1,size=10)
sns.distplot(tmp,color="b",label="midium",hist=False)

#### For large size data
tmp = np.random.normal(loc=0,scale=1,size=50)
sns.distplot(tmp,color="c",label="large",hist=False)

#### For Very large size data
tmp = np.random.normal(loc=0,scale=1,size=5000)
sns.distplot(tmp,color="r",label="very large",hist=False)
plt.show()