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
import pandas as pd 
employee_info = { 'Name' :['rakesh','mamata','shewli','tommy','kronten'],
                 'salary':['90000','65000','50000','78000','NaN'],
                 'age':['23','24','22','NaN','34']}
labels = ['1','2','3','4','5']
df=pd.DataFrame(employee_info,index=labels)
print (df[0:3])
import pandas as pd



# Read a dataset with missing values
flights = pd.read_csv("../input/titanic/train_and_test2.csv")
  # Select the rows that have at least one missing value
flights[flights.isnull().any(axis=1)].head()






import numpy as np
a = np.array([1,2,3,4,5,6,7,8,9])
print("Array 1: ",a)
b = np.array([1,2,3,8,9])
print("Array 2: ",b)
print("Common values between two arrays:")
print(np.intersect1d(a, b))

for i, val in enumerate(a):
    if val in b:
        a = np.delete(a, np.where(a == val)[0][0])
for i, val in enumerate(b):
    if val in a:
        a = np.delete(a, np.where(a == val)[0][0])
print("Arrays after deletion of common elements : ")
print(a)
print(b)





import matplotlib.pyplot as plt
plt.bar(["CSK","KKR","DC","MI"],[180,158,210,125],label="final score",color='y')
plt.legend()
plt.xlabel(' Name of teams ')
plt.xlabel(' Scores ')
plt.title(' Teams vs Scores ')
plt.show