import numpy as np # linear algebra

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

df = pd.read_csv('../input/train.csv')

#print(df.columns.values)

user_data = []

length = len(df['id'])

X, Y = [], []

D = max(df['loss'].values)-min(df['loss'].values)

for i in range(0, length):

    tot = 0.0

    for j in range(1, 15):

        column = 'cont'+str(j)

        avg = df[column].mean()

        tot += abs(df[column].values[i]-avg)

    avg = tot/14

    X.append(avg)

    Y.append(df['loss'].values[i]/D)

    #print(X[i], Y[i])

plt.plot(X, Y, 'b')
for i in range(1, 15):

    field = 'cont'+str(i)

    print(max(df[field].values), min(df[field].values))
print(max(df['loss'].values), min(df['loss'].values), df['loss'].mean())
cnt = df['cat2'].value_counts()

print(cnt['A'])