# ramdom problem to solve !! 

import pandas as pd

df1 = pd.DataFrame([['tom', 10, 0], ['nick', 15,2], ['juli', 14,1]] , columns = ['Name', 'Age', 'col3'])

# df0 = df1.T

list1 = [None] * len(df1)



def Df1ToDf2():

    for i in range(len(df1)): 

        list1[i] = df1[:].values[i] # values might be changed to preserve the datatype

    return list1



print(Df1ToDf2())