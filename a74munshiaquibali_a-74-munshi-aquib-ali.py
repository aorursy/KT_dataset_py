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
import numpy as np
a=np.array([1,2,3])
print(a)
import numpy as np
x=np.array([[1,2,3],[6,8,9]],np.int32)
print(type(x))
print(x.shape)
print(x.dtype)
import numpy as np
bool_arr=np.array([1,0.2,0,None,'b',True,False],dtype=bool)
print(bool_arr)
import numpy as np
a=np.random.randint(0,5,size=(5,4))
print("a\n")
print(a)
b=(a<3).astype(int)
print("\nb")
print(b)
import numpy as np
a=np.array([0,1,2,3,4,6])
print("array1:",a)
a2=[1,2,3,4]
print("array2:",a2)
print("common elements between two array:")
print(np.intersect1d(a,a2))

import numpy as np
a=np.array([1,2,3,4,6])
print("array to remove:",a)
a2=[1,2,3,4,5,6,7]
print("array:",a2)
for i in a:
    if i in a2:
        a2.remove(i)
print("array:",a2)       
import numpy as np
a=np.array([0,1,2,3,4,1,6])
b=np.array([6,8,9,3,8,1,2])
np.where(a==b)

