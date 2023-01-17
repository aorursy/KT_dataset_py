import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
labels = ['a', 'b', 'c']

my_list = [10, 20, 30]

arr = np.array([10, 20, 30])

d = {'a': 10, 'b': 20, 'c': 100}
pd.Series(my_list, index=labels)
pd.Series(arr, labels)
pd.Series(d)
pd.Series(data = labels)
pd.Series([sum, print, len])
ser1 = pd.Series([1, 2, 3, 4], index = ['USA', 'CHINA', 'FRANCE', 'GERMANY'])
ser1
ser2 = pd.Series([1, 2, 3, 4], index = ['USA', 'CHINA', 'ITALY', 'JAPAN'])
ser2
ser1 + ser2