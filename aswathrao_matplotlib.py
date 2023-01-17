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
import matplotlib.pyplot as plt
import numpy as np
plt.plot([1,2,3,4],[1,4,9,16])
plt.show()
plt.plot([1,2,3,4],[1,4,9,16])
plt.title('Figure 1')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()

plt.figure(figsize = (15,8))
plt.plot([1,2,3,4],[1,4,9,16])
plt.show()
plt.plot([1,2,3,4],[1,4,9,16],"go")
plt.title('Figure 1')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()
x = np.arange(1,5)
y = x ** 3 
plt.plot([1,2,3,4],[1,4,9,16],x,y,'r^')
plt.subplot(1,2,1)
plt.plot([1,2,3,4],[1,4,9,16],'r^')
plt.title('First')

plt.subplot(1,2,2)
plt.plot([1,2,3,4],[1,8,27,64],'g')
plt.title('Second')


plt.suptitle('Graphs')
plt.show()
x = np.arange(1,10)
y = x ** 4 
fig,ax = plt.subplots(nrows = 2, ncols =2, figsize = (6,6))
ax[0,1].plot([1,2,3,4],[1,4,9,16],"go")
ax[1,0].plot(x,y)
ax[0,1].set_title('Plot 1')
ax[1,0].set_title('Plot 2')
x = ['A','B','C','D']
y = [33,55,32,87]

plt.bar(x,y)
plt.title('Bar Graph')
plt.xlabel('Section')
plt.ylabel('Average Marks')
import numpy as np
section = ['A','B','C','D']
boys = [33,55,32,87]
girls = [66,87,34,90]

index = np.arange(4)
width = 0.4

plt.bar(index,boys,width)
plt.bar(index+width,girls,width)
plt.title('Bar Graph')
plt.xlabel('Section')
plt.ylabel('Average Marks')

plt.xticks(index + width/2,section)
x = ['A','B','C','D']
y = [33,55,32,87]

plt.pie(y,labels = x)
plt.title('Pie Chart')
plt.legend()
x = np.random.randn(100)
plt.hist(x,10)
x = [10,20,30,40]
y = [100,200,300,400]

plt.scatter(x,y)
plt.xlim(10,40)
plt.ylim(100,400)