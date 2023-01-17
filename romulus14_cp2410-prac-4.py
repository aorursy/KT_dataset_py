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
def transfer(S,T):

    while not S.is_empty():

        T.push(S.pop())
from collections import deque as dqu

Dlist = [1,2,3,4,5,6,7,8]

D = dqu()

Q = dqu()



for i in Dlist:

    D.append(i)

    

Q.append(D.popleft())

Q.append(D.popleft())

Q.append(D.popleft())

D.append(D.popleft())

Q.append(D.popleft())



Q.append(D.pop())



Q.append(D.popleft())

Q.append(D.popleft())

Q.append(D.popleft())



D.append(Q.popleft())

D.append(Q.popleft())

D.append(Q.popleft())

D.append(Q.popleft())

D.append(Q.popleft())

D.append(Q.popleft())

D.append(Q.popleft())

D.append(Q.popleft())





print(D)
from collections import deque as dqu

Dlist = [1,2,3,4,5,6,7,8]

D = dqu()

S = dqu()



for i in Dlist:

    D.append(i)

    

S.append(D.pop())

S.append(D.pop())

S.append(D.pop())

D.appendleft(D.pop())

S.append(D.pop())

S.append(D.popleft())

S.append(D.pop())

S.append(D.pop())

S.append(D.pop())



D.append(S.pop())

D.append(S.pop())

D.append(S.pop())

D.append(S.pop())

D.append(S.pop())

D.append(S.pop())

D.append(S.pop())

D.append(S.pop())

print(D)