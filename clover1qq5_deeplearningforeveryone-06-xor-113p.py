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
w11 = np.array([-2, -2])

w12 = np.array([2, 2])

w2 = np.array([1, 1])

b1 = 3

b2 = -1

b3 = -1



def MLP(x, w, b):

    y = np.sum(w * x) + b

    if y <= 0:

        return 0

    else:

        return 1

    

def NAND(x1, x2):

    return MLP(np.array([x1, x2]), w11, b1)



def OR(x1, x2):

    return MLP(np.array([x1, x2]), w12, b2)



def AND(x1, x2):

    return MLP(np.array([x1, x2]), w2, b3)



def XOR(x1, x2):

    return AND(NAND(x1, x2), OR(x1,x2))







if __name__ == '__main__':

    for x in [(0,0), (1,0), (0,1), (1,1)]:

        y = XOR(x[0], x[1])

        print("입력 값: " + str(x) + "출력 값: " + str(y))