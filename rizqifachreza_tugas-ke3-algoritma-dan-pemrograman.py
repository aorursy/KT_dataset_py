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
#deret fibonacci

bnykbil = 15

n1 = 0

n2 = 1

count = 0



if bnykbil <= 0:

    print ("error")

elif bnykbil == 1:

    print (n1)

else:

    while count < bnykbil:

        print(n1, end=',')

        nth = n1+n2

        n1 = n2

        n2 = nth

        count += 1
#bilangan faktorial

num = 5

if num == 0:

    print("faktorial dari 0 adalah 1")

else:

    for a in range (1,num):

        num = num*a

    print(num)