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
x = [2,4,6,8]
y = [81,93,91,97]
mx = np.mean(x)
my = np.mean(y)
print("x의 평균값:" ,mx)
print("y의 평균값:", my)
divisor = sum([(i-mx)**2 for i in x])
def top(x, mx, y, my):
    d = 0 
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i]-my)
    return d
dividend = top(x, mx, y, my)
print("분모:", divisor)
print("분자:", dividend)
a = dividend / divisor
b = my - (mx*a)
print("기울기 a = ", a)
print("y절편 b = ", b)
