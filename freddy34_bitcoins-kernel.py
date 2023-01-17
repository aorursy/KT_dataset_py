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
import matplotlib.pyplot as plt

df=pd.read_csv("/kaggle/input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")

df.head()

#first 5 paragraph on the list 
import matplotlib.pyplot as plt

df.Open.plot(kind ="line",grid =True,linewidth = 1,color= "r",alpha = 0.5 ,linestyle = ":",label= "Open")

plt.legend()

plt.title("bitcoins")

plt.show()

#open columns has been createed as a line plot method
df.plot(kind = "scatter",x = "High", y ="Close",color="R",linewidth = 1,grid =True,label ="karsilastirma")

plt.legend(loc = "upper-right")

plt.show()

#close and high columns has been comprasion 
df.Weighted_Price.plot(kind = "hist",grid =True,linewidth = 1,color= "g",alpha = 0.5 ,linestyle = ":",label= "bitcoins")

plt.legend()

plt.xlabel("X")

plt.ylabel("Y")

plt.title("bitcoins-hist")

plt.show()

#the    weighted_price columns has been mmonitored as hist plot          
for key,value in df.items():

        print(key,":",value)

        print("")

 # in df datas  i have ve been monitored  key and value 
for index,value in enumerate(df):

    print(value,":",index)

    print("")

    

    #i have been reach  value and idex in the list 
for index , value  in df [["Low"]][0:3].iterrows():

    print (index,":",value)

    #in lox data i have been reach to for first 3 rows