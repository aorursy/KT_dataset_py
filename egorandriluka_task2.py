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
#1.1-8
import numpy as np
a=np.random.randint(10,size=(3,4))
print(1.1)
print(a)
print(1.2)
print (a[0,1])
print(1.3)
print (a[:,1])
print(1.4)
print(a[:,0] [::2])
b = a.reshape(6,2)
print(1.5)
print (b)
c=a/2
print(1.6)
print (c)
print(1.7)
q=a[:,0]
print(np.max(q))
w=a[:,1]
print(np.max(w))
e=a[:,2]
print(np.max(e))
r=a[:,3]
print(np.max(r))
print(1.8)
x=a[0]
print(np.min(x))
#9 
import numpy as np







