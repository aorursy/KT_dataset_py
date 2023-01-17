import pandas as pd

from pandas import Series,DataFrame



import numpy as np

# get titanic & test csv files as a DataFrame

data = pd.read_csv('../input/train.csv')

data = data.iloc[0:5000,0:]

test_data    = pd.read_csv('../input/test.csv')
header = data.columns.values

header = header[1:]
#---- Define a Function to draw number 

# Row Index starts with 10, so that all numbers have same length



def drawNumber(data, Number):

    print(10, 0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7)

    for i in range(0,28*28,28):

            print(int(10+i/28),data.loc[Number, header[i]],data.loc[Number, header[i+1]],data.loc[Number, header[i+2]],data.loc[Number, header[i+3]],

                 data.loc[Number, header[i+4]],data.loc[Number, header[i+5]],data.loc[Number, header[i+6]],data.loc[Number, header[i+7]],

                 data.loc[Number, header[i+8]],data.loc[Number, header[i+9]],data.loc[Number, header[i+10]],data.loc[Number, header[i+11]],

                 data.loc[Number, header[i+12]],data.loc[Number, header[i+13]],data.loc[Number, header[i+14]],data.loc[Number, header[i+15]],

                 data.loc[Number, header[i+16]],data.loc[Number, header[i+17]],data.loc[Number, header[i+18]],data.loc[Number, header[i+19]],

                 data.loc[Number, header[i+20]],data.loc[Number, header[i+21]],data.loc[Number, header[i+22]],data.loc[Number, header[i+23]],

                 data.loc[Number, header[i+24]],data.loc[Number, header[i+25]],data.loc[Number, header[i+26]],data.loc[Number, header[i+27]])

        



#Change pixel values to 1 if >0

data0_1 = data.copy()





for e in header:

    data0_1.loc[data0_1[e] > 0, e ] = 1
print(drawNumber(data0_1, 0))

print(drawNumber(data0_1, 1))