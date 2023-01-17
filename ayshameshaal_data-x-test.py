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
%ls
[i for i in range(10) if (i%2!=0)]
[i for i in range(10) if (i>1)&(i<6)]
def oddNum(x):

    if not (x%2==0):

        return(True)

    else:

            return(False) 

oddNum(2)



[""+ str(i)+"    "+str(oddNum(i)) for i in range(10)]





a = 3 

b = 5 

print (b**a)