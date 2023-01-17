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
#課題1

storages = np.array([1, 2, 3, 4, 5])

storages += 1

print(storages)
#課題2

arr1 = np.array([1, 2, 3, 4])

arr2 = np.array([6, 7, 8, 9])

print(arr1.dot(arr2) + 4)
#課題3

arr = np.array([[1, 2, 3], 
                [6, 50, 400], 
                [5, 10, 100]])


print(np.average(arr, axis=0))
#課題4

index=["Taro", "Jiro", "Saburo", "Hanako", "Yoshiko"]
data=[90, 100, 70, 80, 100]
series=pd.Series(data, index=index)
print(series[series!=100])