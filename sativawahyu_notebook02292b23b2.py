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
arr = np.array([1,2,3,4,5])
print(arr)
import numpy as np
arr = np.array([1,2,3,4,5])
type(arr)
import numpy as np
arr = np.array([1,2,3,4,5])
arr.dtype
import numpy as np
arr = np.array([1,2,3,4,5])
arr.ndim
import numpy as np
arr = np.array([1,2,3,4,5])
arr.shape
import numpy as np
arr = np.array([1,2,3,4,5])
arr2 = np.array([10,11,12,13,15])
arr+arr2
import numpy as np
arr = np.array([1,2,3,4,5])
arr2 = np.array([10,11,12,13,15])
arr-arr2
import numpy as np
arr = np.array([1,2,3,4,5])
arr2 = np.array([10,11,12,13,15])
arr*arr2
import numpy as np
arr = np.array([1,2,3,4,5])
arr2 = np.array([10,11,12,13,15])
arr/arr2
import numpy as np
arr = np.array([1,2,3,4,5])
np.sin(arr)
import numpy as np
arr = np.array([1,2,3,4,5])
arr[2]
import numpy as np
arr = np.array([1,2,3,4,5])
arr[2] = 100
arr
import numpy as np
arr = np.array([[1,2,3,4,5],
               [5,4,3,2,1]])
arr
import numpy as np
arr = np.array([[1,2,3,4,5],
               [5,4,3,2,1]])
arr.shape
import numpy as np
arr = np.array([[1,2,3,4,5],
               [5,4,3,2,1]])
arr.size
import numpy as np
arr = np.array([1,2,3,4,5])
arr[1:3]
import pandas as pd
arr = pd.Series([1,2,3,4,5])
arr
import pandas as pd
arr = pd.Series([1,2,3,4,5])
type(arr)
import pandas as pd
arr = pd.Series([1,2,3,4,5], index = ['a','b','c','d','e'])
arr
import pandas as pd
arr = pd.Series([1,2,3,4,5], index = ['a','b','c','d','e'])
arr.index
import pandas as pd
arr = pd.Series([1,2,3,4,5], index = ['a','b','c','d','e'])
arr.values
import pandas as pd
arr = pd.Series([1,2,3,4,5], index = ['a','b','c','d','e'])
arr['a']
import pandas as pd
arr = pd.Series([1,2,3,4,5], index = ['a','b','c','d','e'])
arr['a'] = 100
arr
import pandas as pd
arr = pd.Series([1,2,3,4,5], index = ['a','b','c','d','e'])
arr[['a', 'b']]
import pandas as pd
arr = pd.Series([1,2,3,4,5], index = ['a','b','c','d','e'])
arr2 = pd.Series([10,20,30,40,50], index = ['a','b','c','d','e'])
arr+arr2
import pandas as pd
data = { 'Sekolah': [ 'SMK N1', 'SMK N2', 'SMK Nasional' ],
        'Tahun': ['2020', '2020', '2020'],
        'Jumlah Murid': [200,200,300] }
frame = pd.DataFrame(data)
frame
import pandas as pd
data = { 'Sekolah': [ 'SMK N1', 'SMK N2', 'SMK Nasional' ],
        'Tahun': ['2020', '2020', '2020'],
        'Jumlah Murid': [200,200,300] }
frame = pd.DataFrame(data)
type(frame)
import pandas as pd
data = { 'Sekolah': [ 'SMK N1', 'SMK N2', 'SMK Nasional' ],
        'Tahun': ['2020', '2020', '2020'],
        'Jumlah Murid': [200,200,300] }
frame = pd.DataFrame(data)
frame.shape
import pandas as pd
data = { 'Sekolah': [ 'SMK N1', 'SMK N2', 'SMK Nasional' ],
        'Tahun': ['2020', '2020', '2020'],
        'Jumlah Murid': [200,200,300] }
frame = pd.DataFrame(data)
frame.info
import pandas as pd
data = { 'Sekolah': [ 'SMK N1', 'SMK N2', 'SMK Nasional' ],
        'Tahun': ['2020', '2020', '2020'],
        'Jumlah Murid': [200,200,300] }
frame = pd.DataFrame(data)
frame.head
import pandas as pd
data = { 'Sekolah': [ 'SMK N1', 'SMK N2', 'SMK Nasional' ],
        'Tahun': ['2020', '2020', '2020'],
        'Jumlah Murid': [200,200,300] }
frame = pd.DataFrame(data)
frame.tail
import pandas as pd
data = { 'Sekolah': [ 'SMK N1', 'SMK N2', 'SMK Nasional' ],
        'Tahun': ['2020', '2020', '2020'],
        'Jumlah Murid': [200,200,300] }
frame = pd.DataFrame(data)
frame.columns
import pandas as pd
data = { 'Sekolah': [ 'SMK N1', 'SMK N2', 'SMK Nasional' ],
        'Tahun': ['2020', '2020', '2020'],
        'Jumlah Murid': [200,200,300] }
frame = pd.DataFrame(data)
frame.index
import pandas as pd
data = { 'Sekolah': [ 'SMK N1', 'SMK N2', 'SMK Nasional' ],
        'Tahun': ['2020', '2020', '2020'],
        'Jumlah Murid': [200,200,300] }
frame = pd.DataFrame(data)
frame.describe
import pandas as pd
data = { 'Sekolah': [ 'SMK N1', 'SMK N2', 'SMK Nasional' ],
        'Tahun': ['2020', '2020', '2020'],
        'Jumlah Murid': [200,200,300] }
frame = pd.DataFrame(data)
frame.describe
import pandas as pd
data = { 'Sekolah': [ 'SMK N1', 'SMK N2', 'SMK Nasional' ],
        'Tahun': ['2020', '2020', '2020'],
        'Jumlah Murid': [200,200,300] }
frame = pd.DataFrame(data)
frame['Sekolah'].value_counts()
import pandas as pd
data = { 'Sekolah': [ 'SMK N1', 'SMK N2', 'SMK Nasional' ],
        'Tahun': ['2020', '2020', '2020'],
        'Jumlah Murid': [200,200,300] }
frame = pd.DataFrame(data)
frame.loc[1]
import pandas as pd
data = { 'Sekolah': [ 'SMK N1', 'SMK N2', 'SMK Nasional' ],
        'Tahun': ['2020', '2020', '2020'],
        'Jumlah Murid': [200,200,300] }
frame = pd.DataFrame(data)
frame.loc[1:2]
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

arr = np.linspace(0,2*np.pi,100)
cos_x = np.cos(arr)
fig,ax = plt.subplots()
_ = ax.plot(arr, cos_x)
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

arr = np.linspace(0,2*np.pi,100)
cos_x = np.cos(arr)
fig,ax = plt.subplots()
_ = ax.plot(arr, cos_x)
_ = ax.set_aspect('equal')
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

arr = np.linspace(0,2*np.pi,100)
cos_x = np.cos(arr)
fig,ax = plt.subplots()
_ = ax.plot(arr, cos_x, markersize=20, linestyle='-.', color='red', label='cos')
_ = ax.set_aspect('equal')
_ = ax.legend()
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

x = np.array([1,2,3])
y = np.array([4,5,6])

fig, ax = plt.subplots()
_ = ax.scatter(x,y)
_ = ax.set_xlabel('X Axis')
_ = ax.set_ylabel('XY Axis')
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

kat = ['Action', 'Adventure', 'Horror', 'Zombie']
jum = [10, 12, 7, 15]

fig, ax = plt.subplots()
_ = ax.bar(kat, jum)
_ = ax.set_xlabel("Kategori")
_ = ax.set_ylabel("Jumlah")
_ = ax.set_title("Kategori Yang Disukai Wibu")