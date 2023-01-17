import pandas as pd
import numpy as np

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

print('Size of training data: ' + str(df_train.shape))
print('Size of testing data:  ' + str(df_test.shape))

print('\nColumns:' + str(df_train.columns.values))
print('\nColumns:' + str(df_test.columns.values))


g = np.array(df_train)
g.shape

g = g[:,1:]
places = g[:,-1]
places.shape
g = g[:,:4]
g
n = int(len(g)*0.9)
print( n)

train_labels = places[:n]
test_labels = places[n+1 :]

train = g[:n,:]
test = g[n+1,:]
sim = np.dot(test, train.T)
print (sim.shape)
#l = places[n+1]
a = np.argsort(-sim)
n = [train_labels[i] for i in a[:25]]
print(places[int(len(g)*0.9)+1])
n