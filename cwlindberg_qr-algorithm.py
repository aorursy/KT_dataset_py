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
from numpy.linalg import eigh
A = np.array([[1, 4, 8, 4],
            [4, 2, 3, 7],
            [8, 3, 6, 9],
            [4, 7, 9, 2]],
            float)

x, V = eigh(A)
x, V
def mag(a):
    """Calculate the magnitude of vector a"""
  
    mag_value = np.sqrt(np.sum(a**2))

    return mag_value
def calc_q(A):
    """Calculate the u vectors given a matrix A"""
    rows = A.shape[0]
    cols = A.shape[1]
    
    q = np.empty([rows, cols])
    U = []

    for i in np.arange(cols):    

        a = A[:, i]

        for j in np.arange(i+1):

            if j == 0:        
                u = a

            else: 
                u = u - np.dot(q[j-1], a)*q[j-1]

        U = np.append(U, u)


        q[i] = (u/mag(u))

    Q = q.T
    return Q, U
  
A = np.array([[1, 4, 8, 4],
            [4, 2, 3, 7],
            [8, 3, 6, 9],
            [4, 7, 9, 2]],
            float)

Q, U = calc_q(A)
np.dot(Q,Q.T)
# manual version of the same code
A = np.array([[1, 4, 8, 4],
            [4, 2, 3, 7],
            [8, 3, 6, 9],
            [4, 7, 9, 2]],
            float)

a0 = A[:, 0]
u0 = a0
q0 = u0/mag(u0)

a1 = A[:,1]
u1 = a1 - np.dot(q0, a1)*q0
q1 = u1 / mag(u1)

a2 = A[:, 2]
u2 = a2 - np.dot(q0, a2)*q0 - np.dot(q1, a2)*q1

u0, u1, u2
def get_qr(A):
  """Get QR decomposition of matrix A"""

  Q = np.hstack((qs))
  # R = np.zeros(A.shape)
  # for i in range(len(R)):
  #   rowvals    = np.dot(qs[i], A)
  #   R[i,i]     = mag(us[i])
  #   R[i, i+1:] = rowvals[i+1:]
  return Q.T.dot(A).dot(Q), Q

tol = 1e-4
As  = []

