import warnings

warnings.filterwarnings("ignore")

from sklearn.datasets import load_boston

from sklearn import preprocessing

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from prettytable import PrettyTable

from sklearn.linear_model import SGDRegressor

from sklearn import preprocessing

from sklearn.metrics import mean_squared_error

from numpy import random

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_boston

import os

from matplotlib import cm

from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn.preprocessing import StandardScaler

from sklearn.utils import resample

from sklearn.metrics import r2_score

import timeit

print("DONE")
!pip install neurokit2
import neurokit2 as nk

from scipy.io import loadmat

import wfdb



# This function loads ECG recordings and meta data given a specific recording number

def load_challenge_data(filename):

    x = loadmat(filename)

    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat','.hea')

    input_header_file = os.path.join(new_file)

    with open(input_header_file,'r') as f:

        header_data=f.readlines()

    return data, header_data
# This piece of code searches through all data, in 1 out of the 2 ECG databases, after patients with LVH diagnose. 

#Then the second heartbeat in every record is added to a new list and interpolated to a 12 x 500 matrix 



from scipy import interpolate

lvh_data = []

starttime = timeit.default_timer()

print("The start time is :",starttime)

for i in sorted(os.listdir("/kaggle/input/china-12lead-ecg-challenge-database/Training_2/")):

    if i.endswith(".mat"):

        data, header_data = load_challenge_data("/kaggle/input/china-12lead-ecg-challenge-database/Training_2/"+i)

        diagnose = header_data[15][5:-1]

        diagnose = diagnose.split(",")

        diagnose = np.asarray(diagnose)

        if pd.Series('164873001').isin(diagnose).any():

            _, rpeaks = nk.ecg_peaks(data[1], sampling_rate=int(header_data[0].split(" ")[2]))

            split_num = int(((np.diff(rpeaks['ECG_R_Peaks'])[0]+np.diff(rpeaks['ECG_R_Peaks'])[1])/2)/2)

            data = data/int(header_data[1].split(" ")[2].split("/")[0])

            ecg3d_lvh=[]

            for i in range(data.shape[0]):

                ecg3d_lvh.append(data[i][rpeaks['ECG_R_Peaks'][1]-split_num:rpeaks['ECG_R_Peaks'][1]+split_num])

            ecg3d_lvh = np.asarray(ecg3d_lvh)

            x=np.arange(ecg3d_lvh.shape[1])

            y=np.arange(ecg3d_lvh.shape[0])

            #x,y = np.meshgrid(x, y)

            f = interpolate.interp2d(x, y, ecg3d_lvh, kind='cubic')

            xnew = np.linspace(0,len(x),500)

            ynew = np.arange(len(y))

            #xnew,ynew = np.meshgrid(xnew, ynew)

            znew = f(xnew, ynew)

            lvh_data.append(znew)

            print("Time after adding first patient data is :", timeit.default_timer() - starttime)

            print(len(lvh_data))

        else:

            pass

lvh_data = np.asarray(lvh_data)
# This piece of code searches through all data, in 1 out of the 2 ECG databases, after patients with normal sinus rythm. 

#Then the second heartbeat in every record is added to a new list and interpolated to a 12 x 500 matrix 



norm_data = []

starttime = timeit.default_timer()

for i in sorted(os.listdir("/kaggle/input/china-physiological-signal-challenge-in-2018/Training_WFDB/")):

    if i.endswith(".mat"):

        data, header_data = load_challenge_data("/kaggle/input/china-physiological-signal-challenge-in-2018/Training_WFDB/"+i)

        diagnose = header_data[15][5:-1]

        diagnose = diagnose.split(",")

        diagnose = np.asarray(diagnose)

        if pd.Series('426783006').isin(diagnose).any():

            _, rpeaks = nk.ecg_peaks(data[1], sampling_rate=int(header_data[0].split(" ")[2]))

            split_num = int(((np.diff(rpeaks['ECG_R_Peaks'])[0]+np.diff(rpeaks['ECG_R_Peaks'])[1])/2)/2)

            data = data/int(header_data[1].split(" ")[2].split("/")[0])

            ecg3d_norm=[]

            for i in range(data.shape[0]):

                ecg3d_norm.append(data[i][rpeaks['ECG_R_Peaks'][1]-split_num:rpeaks['ECG_R_Peaks'][1]+split_num])

            ecg3d_norm = np.asarray(ecg3d_norm)

            x=np.arange(ecg3d_norm.shape[1])

            y=np.arange(ecg3d_norm.shape[0])

            #x,y = np.meshgrid(x, y)

            f = interpolate.interp2d(x, y, ecg3d_norm, kind='cubic')

            xnew = np.linspace(0,len(x),500)

            ynew = np.arange(len(y))

            #xnew,ynew = np.meshgrid(xnew, ynew)

            znew = f(xnew, ynew)

            norm_data.append(znew)

            print("Time after adding first patient data is :", timeit.default_timer() - starttime)

            print(len(norm_data))

        else:

            pass

norm_data = np.asarray(norm_data)
z = np.mean(norm_data[:100],axis=0)

x = np.arange(500)

y = np.arange(12)

x,y=np.meshgrid(x,y)
fig = plt.figure(figsize=(32,12))

ax = fig.gca(projection ='3d')

surf = ax.plot_surface(x,y,z,cmap=cm.bone, linewidth = 0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))

ax.view_init(20, 75)

ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf,shrink=0.5, aspect=5)

fig.show()
def make_design_matrix(x,y,n):

    x = x.ravel()

    y = y.ravel()

    if len(x) != len(y):

        print("x and y needs to have the same length!")

        

    X = np.ones(len(x)).reshape(len(x),1)

    for i in range(n):

        if i == 0:

            X_temp = np.hstack(((x**(i+1)).reshape(len(x),1) , (y**(i+1)).reshape(len(y),1)))

            X = np.concatenate((X,X_temp),axis=1)

        else:

            X_temp = np.hstack(((x**(i+1)).reshape(len(x),1) , (y**(i+1)).reshape(len(y),1),((x**i) * (y**i)).reshape(len(y),1) ))

            X = np.concatenate((X,X_temp),axis=1)



    return X
def predict(x,y,n,beta):

    pred = beta[0]

    for i in range(n):

        if i == 0:

            pred = pred + beta[1] * x + beta[2] * y

        else:

            pred = pred + beta[i*3] *(x**(i+1)) + beta[(i*3)+1]*(y**(i+1)) + beta[(i*3)+2]*((x**i) * (y**i))

    return pred
# This function calculates beta for OLS

def calc_beta(X,y):

    beta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    return beta


z = np.mean(norm_data[:100],axis=0)

x = np.arange(500)

y = np.arange(12)

x,y=np.meshgrid(x,y)





polynomial = 50



x_and_y=np.hstack((x.ravel().reshape(x.ravel().shape[0],1),y.ravel().reshape(y.ravel().shape[0],1)))

scaler = StandardScaler()

scaler.fit(x_and_y)

x_and_y_scaled = scaler.transform(x_and_y)









for poly in range(polynomial):







    X_train = make_design_matrix(x_and_y_scaled.T[0],x_and_y_scaled.T[1],poly+1)

    beta = calc_beta(X_train, z.ravel())

    pred=predict(x_and_y_scaled.T[0],x_and_y_scaled.T[1],poly+1,beta)









    

    print('Polynomial degree:', poly+1)

    print('Error:', mean_squared_error(z.ravel(),pred))

    print("R2-scores:",r2_score(z.ravel(),pred))

        

    #X_plot_scaled = scaler.transform(X)    

    z_pred_for_plot = predict(x_and_y_scaled.T[0],x_and_y_scaled.T[1],poly+1,beta)

    fig = plt.figure(figsize=(32,12))

    ax = fig.gca(projection ='3d')

    surf = ax.plot_surface(x,y,z_pred_for_plot.reshape(x.shape[0],x.shape[1]),cmap=cm.bone, linewidth = 0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))

    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf,shrink=0.5, aspect=5)

    ax.view_init(20, 75)

    fig.suptitle("A {} degree polynomial fit of Franke function using OLS and K-fold crossval".format(poly+1) ,fontsize="40", color = "black")

    fig.show()
X_train = make_design_matrix(x_and_y_scaled.T[0],x_and_y_scaled.T[1],21)

beta = calc_beta(X_train, z.ravel())

z_pred_for_plot = predict(x_and_y_scaled.T[0],x_and_y_scaled.T[1],21,beta)

fig = plt.figure(figsize=(32,12))

ax = fig.gca(projection ='3d')

surf = ax.plot_surface(x,y,z_pred_for_plot.reshape(x.shape[0],x.shape[1]),cmap=cm.bone, linewidth = 0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))

ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf,shrink=0.5, aspect=5)

ax.view_init(20, 75)

fig.suptitle("A {} degree polynomial fit of Franke function using OLS and K-fold crossval".format(21) ,fontsize="40", color = "black")

fig.show()
for i in range(10):

    print(1/(10**(i+1) ))
1/(10*2)
def learning_schedule(start_lr,t):

    lr = start_lr/(10**(t+1))

    return lr
z = np.mean(norm_data[:100],axis=0)

x = np.arange(500)

y = np.arange(12)

x,y=np.meshgrid(x,y)

x_and_y=np.hstack((x.ravel().reshape(x.ravel().shape[0],1),y.ravel().reshape(y.ravel().shape[0],1)))



iterations =int(6000/100)

polynoials=13

number_of_epochs = int(100/10)



best_mse = 0

prev_mse = 1

best_epoch = 0

best_poly = 0

best_iter = 0







scaler = StandardScaler()

scaler.fit(x_and_y)

x_and_y_scaled = scaler.transform(x_and_y)



for pol in range(polynoials):

    for j in range(iterations):

        for k in range(number_of_epochs):

            n_iter = (j+1)*100

            X=make_design_matrix(x_and_y_scaled.T[0],x_and_y_scaled.T[1],pol)

            theta = np.random.randn(X.shape[1],1)

            n_epochs = j *10

            for epoch in range(n_epochs):

                for i in range(n_iter):

                    epoch = epoch

                    random_index = np.random.randint(n_iter)

                    xi = X[random_index:random_index+1]

                    zi = z.ravel()[random_index:random_index+1]

                    gradients = 2.0/m * xi.T @ ((xi @ theta)-zi)

                    eta = learning_schedule(1,i+n_iter*epoch)

                    theta = theta - eta*gradients

                    #print("learning rate:",eta)

                    

            ypredict = X.dot(theta)



            if (mean_squared_error(z.ravel(),ypredict) < prev_mse):

                prev_mse = mean_squared_error(z.ravel(),ypredict)

                best_mse = mean_squared_error(z.ravel(),ypredict)

                best_poly = pol

                best_iter = n_iter

                best_epoch = epoch

                print("MSE:",mean_squared_error(z.ravel(),ypredict))

                print("R2-score:",r2_score(z.ravel(),ypredict))
X=make_design_matrix(x,y,best_poly)

theta = np.random.randn(X.shape[1],1)

for epoch in range(best_epoch):

    for i in range(best_iter):

        random_index = np.random.randint(n_iter)

        xi = X[random_index:random_index+1]

        zi = z.ravel()[random_index:random_index+1]

        gradients = 2.0 * xi.T @ ((xi @ theta)-zi)

        eta = learning_schedule(1,i+best_iter*epoch)

        theta = theta - eta*gradients

print("theta from own sdg")

print(theta)

ypredict = X.dot(theta)
print("MSE:",mean_squared_error(z.ravel(),ypredict))

print("R2-score:",r2_score(z.ravel(),ypredict))

fig = plt.figure(figsize=(32,12))

ax = fig.gca(projection ='3d')

surf = ax.plot_surface(x,y,ypredict.reshape(x.shape[0],y.shape[1]),cmap=cm.bone, linewidth = 0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))

ax.view_init(20, 75)

ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf,shrink=0.5, aspect=5)

fig.show()