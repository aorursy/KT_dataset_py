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
cacah=10

bil1=0

bil2=1

i=1

jmlprev=0

for i in range(1,cacah+1,1):

    print("Deret Fibonacci=")

    print(bil1)

    jmlprev=bil1+bil2

    bil1=bil2

    bil2=jmlprev
cacah=15

#Bilangan Pertama

bil1=0

#Bilangan Kedua

bil2=1



count = 2

if cacah <=0:

    print("angka harus lebih besar dari 0")

elif cacah==1:

    print("Deret Fibonacci:",cacah)

    print (bil1)

else:

    print("Deret Fibonacci:",cacah,":")

    print("Bilangan pertama:",bil1,",","Bilangan Kedua:",bil2)

    while count < cacah:

        jmlprev = bil1 + bil2

        print (bil1)

        bil1=bil2

        bil2=jmlprev

        count +=1