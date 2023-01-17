# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
x=['abc','xyz','pqr',[1,2,3,'data','machine leanring','analytics']]

# write a code to print following outputs¶

# q1:  data reverse i.e. atad

q1=x[3][3][-1]+x[3][3][-2]+x[3][3][-3]+x[3][3][-4]

print("q1  : ", q1)

# q2:  machine analytics

t=x[3][4].split()

q2=t[0]+" "+ x[3][-1]

print("q2  : ", q2)

# q3:  1+2+3

q3=str(x[3][0])+"+"+str(x[3][1])+"+"+str(x[3][2])

print("q3  : ", q3)

# q4:  pqrxyz

q4=x[2]+x[1]

print("q4  : ", q4)

# q5:  analyticsatad

q5=x[3][-1]+q1

print("q5  : ", q5)

# q6:  DMLA(First letter of data , machine leanring, analytics)

q6=x[1][0]+x[2][0]+x[0][0]

print("q6  : ", q6)

# q7:  321

q7=str(x[3][2])+str(x[3][1])+str(x[3][0])

print("q7  : ", q7)

# q8:  pqrdatalearning

q8=x[2]+x[3][3]+t[1]

print("q8  : ", q8)

# q9:  xpa

q9=x[0][0]+x[2][0]+x[2][1]+x[0][2]

print("q9  : ", q9)

# q10: apqc

q10=x[3][3][0]+ x[3][4][0]+t[1][0]+x[3][5][0]

print("q10 : ", q10.upper())
import numpy as np

x=np.random.randint(0,10,(5,5))

np.random.seed(0)

x
import numpy as np

q1=x[2:4, 2:4]

q1

q2=x[4:4, 2:4]

q2

q3=x[5:4, 2:4]

q3

q4=x[1:, 2:4]

q4

# q5:  Rows : 2:4  columns : 2:4  and squar all the values

q5=(x[2:4, 2:4])**2

q5



#q6:  Rows : 2:4  columns : 2:4  and divide the values with highest number

q6=(x[2:4, 2:4])/(np.max(x))

q6

#q7:  Rows : 2:4  columns : 2:4  and divide the values with least number 

q7=(x[2:4, 2:4])/(np.min(x))

q7
# q8:  Rows : 2:4  columns : 2:4  and calcualte column level , rowlevel , sum, mean, median , mode

q1

csum=np.sum(q1,axis=0)

csum

rsum=np.sum(q1,axis=1)

rsum

cmean = np.sum(csum)/csum.size

cmean

rmean = np.sum(rsum)/rsum.size

rmean

cmedian=np.median(q1,axis=0)

cmedian

rmedian=np.median(q1,axis=1)

rmedian
import scipy

from scipy import stats

q1

column_mode = stats.mode(q1, axis=0) 

print("column_mode", column_mode)

print()

row_mode = stats.mode(q1, axis=1)

print("row_mode", row_mode)