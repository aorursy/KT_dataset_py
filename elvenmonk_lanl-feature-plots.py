import pandas as pd

import numpy as np

from scipy import stats

from sklearn.preprocessing import StandardScaler



import matplotlib.pyplot as plt

from tqdm import tqdm

plt.rcParams['figure.figsize'] = [25, 625]

import json



X_test = pd.read_csv('../input/lanl-feature-transform/test_features.csv', index_col=[0])

X_tr = pd.read_csv('../input/lanl-feature-transform/train_features.csv', dtype=np.float64)

Y_tr = pd.read_csv('../input/lanl-ttf-error/time_prediction.csv', dtype=np.float64)

print(X_tr.shape)

print(np.nonzero(X_tr.values == -np.inf))



X_tr = X_tr

Y_tr = Y_tr

print(X_tr.shape)

print(Y_tr.shape)



scaler = StandardScaler()

scaler.fit(X_tr)

X_train_scaled = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)

X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)



Y_scaler = StandardScaler()

Y_scaler.fit(Y_tr)

Y_train_scaled = pd.DataFrame(Y_scaler.transform(Y_tr), columns=Y_tr.columns)

N = Y_tr.shape[0]

T = np.arange(N)

good_columns = {column: stats.pearsonr(X_tr[column], (Y_tr.values[:,0]/Y_tr.values[:,1]).ravel())[0] for column in X_tr.columns if abs(stats.pearsonr(X_tr[column], (Y_tr.values[:,0]/Y_tr.values[:,1]).ravel())[0]) > 0.8}

print(json.dumps(good_columns, indent='\t', separators=(',', ':\t')))

good_columns = {column: stats.pearsonr(X_tr[column], Y_tr.values[:,1].ravel())[0] for column in X_tr.columns if abs(stats.pearsonr(X_tr[column], Y_tr.values[:,1].ravel())[0]) > 0.17}

print(json.dumps(good_columns, indent='\t', separators=(',', ':\t')))

M = len(X_tr.columns)

R2 = (M+1)//2

R3 = (M+2)//3

R5 = (M+4)//5

print(M,R2,R3,R5)
plt.figure(figsize=(25, 3*R5))

for i, column in tqdm(enumerate(X_tr.columns)):

    plt.subplot(R5, 5, i + 1)

    plt.title(column)

    plt.axis([0,1,-3,3])

    plt.scatter(Y_tr.values[37:,0]/Y_tr.values[37:,1], X_train_scaled[column][37:], s=1, c=Y_tr.values[37:,1], cmap='gist_rainbow')

    plt.scatter(np.full_like(X_test_scaled[column], 15), X_test_scaled[column], s=1, c='b')

plt.show()

#plt.savefig('features.png')
plt.figure(figsize=(25, 3*R2))

bounds = [0, 37, 333, 697, 925, 1250, 1457, 1638, 2052, 2255, 2502, 2795, 3078, 3305, 3525, 3903, 4146, 4193]

n = len(bounds) - 1

good_columns2 = X_tr.columns[[103,238]]

print(good_columns2)

#m = len(good_columns)

for j, column in tqdm(enumerate(X_tr.columns)):

    plt.subplot(R2, 2, j + 1)

    plt.title('{0}. {1}'.format(j, column))

    plt.axis([0,1,-3,3])

    for i in range(n):

        c = (Y_tr.values[bounds[i],1]-7)/10

        plt.plot(Y_tr.values[bounds[i]:bounds[i+1],0]/Y_tr.values[bounds[i]:bounds[i+1],1], X_train_scaled[column][bounds[i]:bounds[i+1]], c=plt.cm.gist_rainbow(c))

plt.show()

#plt.savefig('features.png')
plt.figure(figsize=(30, 3*R2))

for i, column in tqdm(enumerate(X_tr.columns)):

    plt.subplot(R2, 2, i + 1)

    plt.title(column)

    plt.axis([0,4200,-2,2])

    plt.plot(T, X_train_scaled[column], T, Y_train_scaled-1)

plt.show()

#plt.savefig('features.png')