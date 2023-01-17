import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16,5)
train = pd.read_csv('../input/train.csv', index_col='row_id')
test = pd.read_csv('../input/test.csv', index_col='row_id')
df = pd.concat([train, test])
del train, test
df.sort_values('accuracy', inplace=True)
df.shape[0]
window = 1000000
step = window//10

stdx = []
stdy = []
acc = []
for i in range(0,df.shape[0]-window,step):
    stdx.append(np.histogram(df[i:(i+window)].x, bins=np.arange(0,10.001, 0.001))[0].std())
    stdy.append(np.histogram(df[i:(i+window)].y, bins=np.arange(0,10.001, 0.001))[0].std())
    acc.append(df[i:(i+window)].accuracy.mean())
plt.axvline(15, color='gray')
plt.axvline(38, color='gray')
plt.axvline(160, color='gray')

plt.semilogx(acc, np.array(stdx))
plt.semilogx(acc, np.array(stdy))
plt.xlabel('accuracy')
plt.ylabel('roughness')
plt.legend(['x', 'y'])