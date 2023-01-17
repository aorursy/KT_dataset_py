import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import random
#Pre-prossesing

df = pd.read_csv('/kaggle/input/cs178-wine-quality/winequality-white.csv')

df = df.dropna()

data = df.values



new_data = []

for i in data:

    [temp] = i 

    new_data.append(list(map(float,list(temp.split(";")))))

new_data = np.array(new_data)



X_train = new_data[:,:-1]

Y_train = new_data[:,-1]
#Training



X_trans = np.transpose(X_train) 



XtX = np.matmul(X_trans,X_train)

XtX_inverse = np.linalg.inv(XtX)

XtY = np.matmul(X_trans,Y_train)



Beta = np.matmul(XtX_inverse,XtY)

print(Beta)
#Testing

Y_cap = np.matmul(X_train,Beta)

Y_cap = np.round(Y_cap)
rmse = (((Y_cap - Y_train)**2).mean())**0.5

print(rmse)



plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')

plt.plot(Y_train,'go',Y_cap,'ro')
alpha = 0.0000001

beta = [random.random() for i in range(11) ]

beta = np.array(beta)

beta


rmse1 = 1000

rmse2 = 1000

rmse3 = 1000



while True:

    Y_cap = np.matmul(X_train,beta)

    temp = Y_cap - Y_train

    temp_trans = np.transpose(temp)

    temp = np.matmul(temp_trans,X_train)



    beta = beta - alpha*temp/m

    #print("Beta",beta)

    rmse = (((Y_cap - Y_train)**2).mean())**0.5

    #print("RMSE =",rmse)

    rmse1 = rmse2

    rmse2 = rmse3

    rmse3 = rmse

    temp_mean = (rmse1 +rmse2 +rmse3 )/3

    if(abs(temp_mean - rmse) < 0.00001):

        break

print("Beta",beta)

print("RMSE =",rmse)
#Testing

Y_cap = np.matmul(X_train,beta)

Y_cap = np.round(Y_cap)
rmse = (((Y_cap - Y_train)**2).mean())**0.5

print(rmse)



plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')

plt.plot(Y_train,'go',Y_cap,'ro')