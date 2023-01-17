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
#importing libraries 



import pandas as pd

import numpy as np 





##constructing a dataframe 



#constructing 2d array 

#a=[]

array = np.arange(0,20).reshape(5,4)

array



#constructing a spreadsheet like dataframe  with this array 

df = pd.DataFrame(array,index = ['r0','r1','r3','r4','r5'],columns=['c1','c2','c3','c4'])

df
#saving to file 

df.to_csv('save.csv',index=False)