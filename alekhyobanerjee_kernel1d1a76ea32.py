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
df=pd.read_csv('/kaggle/input/iris/Iris.csv')
df
df.rename(columns={'SepalLengthCm':'SL','SepalWidthCm':'SW','PetalLengthCm':'PL','PetalWidthCm':'PW'},inplace=True)
df['SL'].skew()  #SEPAL LENGTH follows NORMAL DISTRIBUTION
df['SW'].skew()  #SEPAL WIDTH follows NORMAL DISTRIBUTION
df['PL'].skew()   #PETAL LENGTH follows NORMAL DISTRIBUTION
df['PW'].skew()  #PETAL WIDTH follows NORMAL DISTRIBUTION
import math                            #ND calculates normal distribution provided value x,mean , standard deviation

def ND(x,avg,sd):

    var=float(sd)**2

    denom=(2*math.pi*var)**.5

    num=math.exp(-(float(x)-float(avg))**2/(2*var))

    value=num/denom

    return value
types=df['Species'].unique()        #types consist the name of various species
def predict(SL,SW,PL,PW,types):

    result=[]

    for species in types:

        val_type=df[df['Species']==species].shape[0]/df.shape[0]                    #find probablity of species

        

        SL_mean,SL_std=df[df['Species']==species]['SL'].agg(['mean','std'])         #mean , standard deviation Sepal_length of each species

        val_SL=ND(SL,SL_mean,SL_std)                                                #calculating normal distribution

    

        SW_mean,SW_std=df[df['Species']==species]['SW'].agg(['mean','std'])         #mean , standard deviation Sepal_width of each species

        val_SW=ND(SW,SW_mean,SW_std)                                                #calculating normal distribution

    

        PL_mean,PL_std=df[df['Species']==species]['PL'].agg(['mean','std'])         #mean , standard deviation Petal_length of each species

        val_PL=ND(PL,PL_mean,PL_std)                                                #calculating normal distribution

    

        PW_mean,PW_std=df[df['Species']==species]['PW'].agg(['mean','std'])         #mean , standard deviation Petal_width of each species

        val_PW=ND(PW,PW_mean,PW_std)                                                #calculating normal distribution

        

        result.append(val_type*val_SL*val_SW*val_PL*val_PW)                         #Conditional Probability

    print(types[result.index(max(result))])                                         #printing the specie with highest probability

        
predict(4.7,3.7,2,0.3,types)