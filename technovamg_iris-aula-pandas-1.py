import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
testSeries = pd.Series([5,6,322], name = "numeros")

testSeries
testFrame = pd.DataFrame([[200,100], [300,150], [0,454]], index = ['Jan','Fev','Mar'],columns = ['Despesas','Lucro'])

testFrame
dict = {'Despesas':[200,300,0], 'Lucro':[100,150,454,]}

testFrame = pd.DataFrame(dict, index = ['Jan','Fev','Mar'])

testFrame
testFrame.iloc[0,:]
testFrame.iloc[:,0]
testFrame.loc[:,'Lucro']
for i, row in testFrame.iterrows():

    print("i: " + str(i))

    print(row)
def mult2(x):

    return x*2

testFrame.apply(mult2)
add1 = lambda x:x+1

testFrame.apply(add1)
testFrame.sum(axis = 0)
testFrame.sum(axis = 1)