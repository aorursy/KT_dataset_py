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
list = [2,2,6,6,8,8,10,10,11,11,15,17]
mdy = np.median(list)

print("Median=",mdy)
series = pd.Series(list)

desc = series.describe()

print(desc)
Q1 = desc[4]

Q3 = desc[6]

IQR = Q3-Q1

print(IQR)
low_q = Q1-1.5*IQR

up_q = Q3+1.5*IQR

print("Median= {},Q1= {},Q2= {},IQR= {},Lower= {},Upper= {}".format(mdy,Q1,Q3,IQR,low_q,up_q))