import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
tree=DecisionTreeRegressor()
tribun = pd.DataFrame(np.array([1,2,3,4,5,6,7,8,9,10]) , columns=["tribun"])

fiyat = pd.DataFrame(np.array([100,90,80,70,60,50,40,30,20,10]) , columns=["fiyat"])

seyirciler=pd.concat([tribun,fiyat] , axis=1)

s=seyirciler.copy()

s # dataset created
x = s.iloc[:,0].values.reshape(-1,1)

y = s.iloc[:,1].values.reshape(-1,1)
tree.fit(x,y)

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head = tree.predict(x_)
plt.scatter(x,y,color = "blue")

plt.plot(x_ , y_head , color = "red")

plt.xlabel("Tribunün sahaya yakınlığı")

plt.ylabel("Bilet fiyatı")