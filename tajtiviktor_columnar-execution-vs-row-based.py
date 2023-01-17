# This notebook to demonstrate the difference of a column-based execution with row-based.



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time
# 1,000,000 random integers

df = pd.DataFrame(np.random.randint(0,100,size=1000000), columns=list('A'))
df.head()
df.shape
# new column (+1) by iterating over rows

tmp = []

start = time.time()

for i in range(1000000):

    tmp.append(df.iloc[i,0] + 1)

df['B'] = tmp              

end = time.time()

row_based_execution = end - start

print('Row-based execution took: {0:.1f} seconds'.format(row_based_execution))
df.head()
# new column (+1) by calling a function over a column

start = time.time()

def plusone(col):

    return col + 1

df['C'] = df['A'].apply(plusone)

end = time.time()

column_based_execution_function = end - start

print('Column-based execution with function took: {0:.1f} seconds'.format(column_based_execution_function))
# new column (+1) by using lambda function over a column

start = time.time()

df['D'] = df['A'].map(lambda x: x+1)

end = time.time()

column_based_execution_lambda = end - start

print('Column-based execution using lambda function took: {0:.1f} seconds'.format(column_based_execution_lambda))
gain_function = row_based_execution / column_based_execution_function

gain_lambda = row_based_execution / column_based_execution_lambda

print('Gain using a function: {0:.2f}, using lambda function: {1:.2f}'.format(gain_function, gain_lambda))