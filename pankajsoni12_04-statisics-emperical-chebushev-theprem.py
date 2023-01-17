import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
tmp = np.random.normal(loc=20,scale=5,size=400)
plt.figure(figsize=(16,8))
# print(tmp)
plt.subplot(2,2,1)
sns.distplot(tmp,color="m",hist=True)
plt.vlines(20,0.0,0.07,color="r",label="mean(μ)") # Line for Mean 
plt.vlines(15,0.0,0.07,color="g",label="1σ") # Line for 1-sigma 
plt.vlines(25,0.0,0.07,color="g") # Line for 1-sigma
plt.vlines(10,0.0,0.07,color="b",label="2σ") # Line for 2-sigma 
plt.vlines(30,0.0,0.07,color="b") # Line for 2-sigma
plt.vlines(5,0.0,0.07,color="k",label="3σ") # Line for 3-sigma 
plt.vlines(35,0.0,0.07,color="k") # Line for 3-sigma
plt.xticks(range(0,50,5))
plt.legend()
plt.show()
arr = np.array(tmp)
l1 = len(arr)
l2 = len(arr[(arr>=15) & (arr<=25)])
l2/l1
arr = np.array(tmp)
l1 = len(arr)
l2 = len(arr[(arr>=10) & (arr<=30)])
l2/l1
arr = np.array(tmp)
l1 = len(arr)
l2 = len(arr[(arr>=5) & (arr<=35)])
l2/l1
tmp2 = np.random.normal(loc=20,scale=50,size=200)
plt.figure(figsize=(16,8))
# print(tmp2)
plt.subplot(2,2,1)
sns.distplot(tmp2,color="m",hist=True)
plt.vlines(20,0.0,0.007,color="r",label="mean(μ)") # Line for Mean 
plt.vlines(-30,0.0,0.007,color="g",label="1σ") # Line for 1-sigma 
plt.vlines(70,0.0,0.007,color="g") # Line for 1-sigma
plt.vlines(-80,0.0,0.007,color="b",label="2σ") # Line for 2-sigma 
plt.vlines(120,0.0,0.007,color="b") # Line for 2-sigma
plt.vlines(-130,0.0,0.007,color="k",label="3σ") # Line for 3-sigma 
plt.vlines(170,0.0,0.007,color="k") # Line for 3-sigma
plt.xticks(range(-200,200,10),rotation=90)
# plt.legend()
plt.show()
arr2 = np.array(tmp2)
l1 = len(arr2)
l2 = len(arr2[(arr2>=-80) & (arr2<=120)])
l2/l1
arr2 = np.array(tmp2)
l1 = len(arr2)
l2 = len(arr2[(arr2>=-130) & (arr2<=170)])
l2/l1
arr2 = np.array(tmp2)
l1 = len(arr2)
k = 6 # so range of data will be within (20-6*50 and 20+6*50) = 20-300 and 20*300 = -280 to 320
l2 = len(arr2[(arr2>=-280) & (arr2<=320)])
print(1-(1/8)) # 1-(1/k)
print(l2/l1)