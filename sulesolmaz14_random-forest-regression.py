# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/decision-tree-dataset/decision_tree_dataset.csv", sep=";", header=None)
df.head()

#x ve y axes
x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)
#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x,y)

print("Price of tribun 7.8: ", rf.predict([[7.8]]))
x_=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head=rf.predict(x_)

#Visualize
plt.scatter(x,y, color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("Tribun Level")
plt.ylabel("Price")
plt.show()
