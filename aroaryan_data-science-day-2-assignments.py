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
#assignment 1

def lst_squareroot (ar):

    arr=[]

    for i in ar:

        arr.append(i**2)

    return arr

lst_squareroot( [1,2,3,4,5])
#assignment 2

def join_str (ar):

    arr=[]

    n = len(ar)

    #print(n)

    

    if(n%2==0):

        for i in range(0,n,2):

            arr.append(ar[i]+ar[i+1])

    else:

        for i in range(0,n-1,2):

            arr.append(ar[i]+ar[i+1])

        arr.append(ar[n-1])

        

        

    return arr

    

join_str(   ['abc','def', 'ghi','jkl', 'mno'] )