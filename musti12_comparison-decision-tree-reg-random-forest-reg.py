# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.DataFrame({"bilet_price":[100,80,70,60,50,40],

                 "tribün_level":[10,25,15,5,45,35]})

print(df)
new_index=[1,2,3,4,5,6]

df["new_index"]=new_index

df=df.set_index("new_index")

print(df)
df.info()
print(df.describe())
dec_tree=DecisionTreeRegressor(random_state=42)

rf=RandomForestRegressor(n_estimators=100,random_state=42)
x=df.iloc[:,1].values.reshape(-1,1)

y=df.iloc[:,0].values.reshape(-1,1)
dec_tree.fit(x,y)

rf.fit(x,y)
r2_dec=dec_tree.predict(x)

r2_rf=rf.predict(x)



print("r2 square for decision tree reg.:", r2_score(y,r2_dec))

print("r2 square for random forest reg.:", r2_score(y,r2_rf))
plt.scatter(x,y)

plt.xlabel("tribun level")

plt.ylabel("bilette price")

plt.show()
x__=np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head_dec=dec_tree.predict(x__)

y_head_rf=rf.predict(x__)
plt.plot(x__,y_head_dec,label="decision tree reg.")

plt.plot(x__,y_head_rf,label="random forest reg.")

plt.xlabel("tribün level")

plt.ylabel("bilet price")

plt.legend()

plt.show()