# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/random-linear-regression/train.csv')

df = df.dropna()

data = df.values

data
# Training

X_train = data[:,0]

Y_train = data[:,1]

X_train
'''k = 15

Y_cap = []

temp = data.copy()

for item_x in X_train:

    print(item_x)

    temp[:,0] = abs(temp[:,0] - item_x)

    temp2 = temp.copy()

    print(np.argpartition(temp2, k,axis=0)[:k])

    knn = temp2[np.argpartition(temp2, k,axis=0)][:k,1].mean()

    print(knn)

    Y_cap.append(knn)

    temp = data.copy()

    break

#print(Y_cap)

'''



Y_cap = []

temp = X_train.copy()

temp3 = Y_train.copy()

#print(temp)

for item_x in X_train:

    #print(item_x)

    #temp[:] = abs(temp[:] - item_x)

    temp2 = temp.copy()

    sum = 0.0

    for i in range (0,8):

        t1=temp2[temp2 >= item_x].min()

        print(t1)

        t2=temp2[temp2 <= item_x].max()

        print(t2)

        ind_t1 = np.where(temp2 == t1)

        ind_t2 = np.where(temp2 == t2)

        print(ind_t1[0][0])

        print(ind_t2[0][0])

        #break

        sum += temp3[ind_t1[0][0]]

        sum += temp3[ind_t2[0][0]]

        temp2 = np.delete(temp2,ind_t1[0][0])

        if(len(temp2)>ind_t2[0][0]):

            temp2 = np.delete(temp2,ind_t2[0][0])

    break

    Y_cap = np.insert(Y_cap,len(Y_cap),(sum/16))

print(Y_cap)

print(Y_cap.shape)
rmse = (((Y_cap - Y_train)**2).mean())**0.5

print(rmse)



plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')

plt.plot(X_train, Y_train,'go',X_train, Y_cap,'ro')