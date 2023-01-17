try:

    num=int(input("Enter a Number :"))    

    fact=1

    if(num>=0):

        if(num%2==0):

            for i in range(1,num+1):

                fact=fact*i

            print("factorila of ",num," =",fact)

        else:

            for j in range(1,num+1):

                fact=1

                for k in range(1,j+1):

                    fact=fact*k

                print("factorila of ",j," =",fact)

    else:

        print("please Enter a Possitive Number")

except Exception as e:

    print("Enter a Valid Number")

    print(e)





        



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