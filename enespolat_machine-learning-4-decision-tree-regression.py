# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/decision-tree-regression-dataset.csv',header=None)
data.head()
x=data.iloc[:,0].values.reshape(-1,1)
y=data.iloc[:,1].values.reshape(-1,1)
x,y

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
tree_reg=DecisionTreeRegressor()
tree_reg.fit(x,y)
y_head=tree_reg.predict(x)
y_head
#visualize
import numpy as np
plt.scatter(x,y,color='red')
#plt.plot(x,y_head,color='green')
x_=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head2=tree_reg.predict(x_)
plt.plot(x_,y_head2,color='blue')
plt.xlabel('tribun level')
plt.ylabel('ucret')
plt.show()