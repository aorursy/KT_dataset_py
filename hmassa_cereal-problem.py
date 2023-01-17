# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
A = np.matrix([[25,0,1],[20,1,2],[40,1,6]])
print(A)


B = np.array([[110], [110], [210]])
print(B)


# Find Ax=b
np.linalg.inv(A)
X = np.linalg.inv(A)*B

print(X)


# Find correct grams of fat in each
X_star = [[4],[9],[4]]
Fat = np.linalg.inv(A)*X_star
print(Fat)
#Find new A
C = np.matrix([[25,15,10,0,1],[20, 12, 8, 1, 2],[40, 30, 10, 1, 6],[30, 15, 15, 0, 3],[35, 20, 15, 2, 4]])
new_C = np.matrix(C[:,1:5])

b = np.array([[104],[97],[193],[132],[174]])
y = (np.matrix(np.transpose(new_C)*new_C))
Y = np.linalg.inv(y)
w = Y*np.transpose(new_C)*b
print(w)

error = b - new_C*w
print(error)