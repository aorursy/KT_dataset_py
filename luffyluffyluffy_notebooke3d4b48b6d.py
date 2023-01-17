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
st = "***"

st
for i in range(6, -1, -1):

    s = ""

    for j in range(i):

        s = s+(" ")

    s=s + (st)

    for j in range((6-i)*2):

        s = s+(" ")

    s=s + (st)



    print(s)

    



    

for i in range(1, 7, 1):

    s = ""

    for j in range(i):

        s = s+(" ")

    s=s + (st)

    for j in range((6-i)*2):

        s = s+(" ")

    s=s + (st)



    print(s)

    
import random

base = float(input())

y = 0.05 * base





high = random.uniform(-y, y)

low = random.uniform(-y, high)

Open = random.uniform(low, high)

close = random.uniform(low, high)





high += base

low += base

Open += base

close += base

print("high :", high)

print("low :", low)

print("Open :", Open)

print("close :", close)
import random

base = float(input())

y = 0.05 * base



n = int(input())





for i in range(n):

    high = random.uniform(0, y)

    low = random.uniform(0, high)

    Open = random.uniform(low, high)

    close = random.uniform(low, high)





    high += base

    low += base

    Open += base

    close += base



    print("case :", i+1, " \n")

    print("high :", high)

    print("low :", low)

    print("Open :", Open)

    print("close :", close)

    print("\n")