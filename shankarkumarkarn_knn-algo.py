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
apti=[8,9,7,2,1,4,8,9,10]

comm=[3,10,2,5,3,9,8,7,2]

label=['I','L','I','A','P','S','L','L','I']

dist=[0,0,0,0,0,0,0,0,0]

apti10=int(input("Enter the apti marks of the new student"))

comm10=int(input("Enter the communication skill marks of the new student"))



for i in range(0,9):

    dist[i]=((apti[i]-apti10)**2+(comm[i]-comm10)**2)**0.5

min=0

for i in range(0,9):

    if(dist[i]<dist[min]):

        min=i

        

label10=label[min] 

print("label of student 10 is"+' '+label10)