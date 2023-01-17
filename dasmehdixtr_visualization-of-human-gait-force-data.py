# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
def datanormalizer(data):

    a = 1

    b = 0

    c = 0

    newData = pd.DataFrame(index=np.arange(0,60),columns=['FP1_x','FP2_x','FP1_y','FP2_y','FP1_z','FP2_z'])

    for each in data.index:

        a = a+1

        if a%1000 == 0:

            newData.iloc[c] = data.iloc[b:a,:].sum()/1000

            b = a

            c = c+1

    return newData

data = pd.read_csv('/kaggle/input/human-gait-phase-dataset/data/GP1_0.6_force.csv')

data.head()
timer = np.arange(0,60000)

plt.scatter(timer,data['FP1_x'],color="red")

plt.scatter(timer,data['FP2_x'],color="blue")

plt.xlabel('mS(time)')

plt.ylabel('Force')

plt.title('x-axis force change relative to time')

plt.grid()

plt.show()
plt.scatter(timer,data['FP1_y'],color="red")

plt.scatter(timer,data['FP2_y'],color="blue")

plt.xlabel('mS')

plt.ylabel('Force')

plt.title('y-axis force change due to time')

plt.grid()

plt.show()
plt.scatter(timer,data['FP1_z'],color="red")

plt.scatter(timer,data['FP2_z'],color="blue")

plt.xlabel('mS')

plt.ylabel('Force')

plt.title('z-axis force change due to time')

plt.grid()

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111,projection='3d')

ax.scatter(data['FP1_x'],data['FP1_y'], data['FP1_z'], color='red', marker='o')

ax.scatter(data['FP2_x'],data['FP2_y'], data['FP2_z'], color='blue', marker='.')

ax.set_xlabel('X Label')

ax.set_ylabel('Y Label')

ax.set_zlabel('Z Label')

plt.title("3D Left-Right Foot Point Cloud")

plt.show()
q = datanormalizer(data=data)



plt.scatter(np.arange(0,60),q['FP1_x'])

plt.scatter(np.arange(0,60),q['FP2_x'])

plt.title('reduced data')

plt.xlabel('S')

plt.ylabel('x_basinc')

plt.show()