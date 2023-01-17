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
# Big endian to little endian converter 



def little(string):

    t = bytearray.fromhex(string)

    t.reverse()

    return ''.join(format(x, '02x') for x in t).lower()
#Panda frame



# print(os.listdir("../input/utxo-set/results.txt"))



colnames = ['I/O', 'TxID', 'Vector', 'Timestamp']

data = pd.read_csv("../input/utxo-set/results.txt",names=colnames, sep=",", header=None, nrows=1000000, engine="python")



result = data.fillna(axis=0, method='ffill')

print(result)
# table1



# Drop these row indexes from dataFrame

indexNames = result[result['TxID'] == '0000000000000000000000000000000000000000000000000000000000000000'].index

indexNames2 = result[result['I/O'] == 'output'].index

result.drop(indexNames, inplace=True)

result.drop(indexNames2, inplace=True)

result['littleTxID'] = result.apply(lambda x: little(x.TxID), axis=1)
# table 2



result2 = data.fillna(axis=0, method='ffill')

indexNames = result2[result2['TxID'] == '0000000000000000000000000000000000000000000000000000000000000000'].index

indexNames2 = result2[result2['I/O'] == 'input'].index

result2.drop(indexNames, inplace=True)
resultCombo = pd.merge(result, result2, how='left', left_on='littleTxID', right_on='TxID')

resultCombo['UTXO_Spent_ElapsedTime'] = resultCombo['Timestamp_x'] - resultCombo['Timestamp_y']

resultCombo['UTXO_Spent_ElapsedTime_Days'] = resultCombo['UTXO_Spent_ElapsedTime']/86400
resultCombo[['UTXO_Spent_ElapsedTime_Days']].plot(kind='hist', bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])