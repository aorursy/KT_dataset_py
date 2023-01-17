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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from matplotlib import pyplot as plt
import numpy as np
x = [4,7,3,9,1,6]
y1 = [9,1,2,3,4,9]
y2 = [4,7,8,1,2,6]

plt.scatter(x,y1)
plt.scatter(x,y2, color="r")
plt.grid(True)
iris=pd.read_csv("../input/iris/Iris.csv")
iris.head()
x = np.arange(1,11)
x
y1 = 2*x
y2 = 3*x
y1,y2
plt.plot(x,y1,color='red',linewidth=2)
plt.plot(x,y2,color='blue', linewidth=2)

plt.title('Line Plot')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.grid(True)

plt.show()
Players = {'Sachin':97, 'Virat':87, 'Dhoni':71, 'Dhawan':92}
Batsman = list(Players.keys())
Runs_Scored = list(Players.values())
Batsman, Runs_Scored
plt.bar(Batsman,Runs_Scored, color='purple')
plt.title('India Batting')
plt.xlabel('Player')
plt.ylabel('Runs')
plt.grid(True)
plt.show()
plt.hist(iris['SepalLengthCm'], bins=20)

plt.show()
plt.hist(iris['PetalWidthCm'], bins=20)

plt.show()
plt.hist(iris['SepalWidthCm'], bins=20)

plt.show()
iris.boxplot(column='SepalLengthCm', by='Species')

iris.boxplot(column='PetalWidthCm', by='Species')
import seaborn as sns
sns.boxplot(x=iris['Species'], y=iris['SepalLengthCm'])
Monthly_Expenditure = ['Rent', 'Travel', 'Food', 'Extra']
Expenditure = [4500,3600,3300,2500]
plt.figure(figsize=(6,6))
plt.pie(Expenditure,labels=Monthly_Expenditure, autopct='%0.1f%%', shadow=True)
plt.show()
