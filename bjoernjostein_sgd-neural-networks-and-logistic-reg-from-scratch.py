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
def make_X_matrix_new(x,y,n):

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
def predict_new(x,y,n,beta):

    pred = beta[0]

    for i in range(n):

        if i == 0:

            pred = pred + beta[1] * x + beta[2] * y

        else:

            pred = pred + beta[i*3] *(x**(i+1)) + beta[(i*3)+1]*(y**(i+1)) + beta[(i*3)+2]*((x**i) * (y**i))

    return pred
def FrankeFunction(x,y):

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))

    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))

    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))

    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    noise = np.random.normal(0, 0.1, len(x)*len(x)) 

    noise = noise.reshape(len(x),len(x))

    return term1 + term2 + term3 + term4 + noise



def xy_data(n):

    x = np.linspace(0,1,n)

    y = np.linspace(0,1,n)

    x,y = np.meshgrid(x,y)

    return x,y
# This function calculates beta for OLS

def calc_beta(X,y):

    beta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    return beta
# K fold algorithm inspired by:

# https://machinelearningmastery.com/implement-resampling-methods-scratch-python/



from random import seed

from random import randrange

 

# Split a dataset into k folds

def cross_validation_split(dataset, folds=10):

    dataset_split = list()

    dataset_copy = list(dataset)

    fold_size = int(len(dataset) / folds)

    for i in range(folds):

        fold = list()

        while len(fold) < fold_size:

            index = randrange(len(dataset_copy))

            fold.append(dataset_copy.pop(index))

        dataset_split.append(fold)

    return dataset_split
# Make data set

n = 40

x,y = xy_data(n)

z = FrankeFunction(x,y)





# number of k-folds

k_folds = 10



# Polynomial fit

polynomial = 25



# Stacking x and y 

x_and_y=np.hstack((x.ravel().reshape(x.ravel().shape[0],1),y.ravel().reshape(y.ravel().shape[0],1)))



# Scaling data

scaler = StandardScaler()

scaler.fit(x_and_y)

x_and_y_scaled = scaler.transform(x_and_y)



# Make list and arrays to store results

all_r2_ols_cv=[]

mean_r2_ols_cv=[]

error = np.zeros(polynomial)

bias = np.zeros(polynomial)

variance = np.zeros(polynomial)

polydegree = np.zeros(polynomial)

train_error = np.zeros(polynomial)





for poly in range(polynomial):

    # Make list and arrays to store results

    r2_ = []

    

    #Make array to store predictions

    pred_test = np.empty((int(z.ravel().shape[0]*(1/k_folds)), k_folds))

    pred_train = np.empty((int(z.ravel().shape[0]*(1-(1/k_folds))), k_folds))

    

    # Stacking x , y (X) and z 

    #data = np.hstack((x_and_y_scaled,z.ravel().reshape(n**2,1)))

    data = np.hstack((x_and_y_scaled,z.ravel().reshape(z.shape[0]*z.shape[1],1)))

    

    #Make folds 

    folds = cross_validation_split(data, k_folds)

    for i in range(k_folds):

        #Make train and test data using the i'th fold

        n_fold = folds.copy()

        test_data = n_fold.pop(i)

        test_data= np.asarray(test_data)

        train_data = np.vstack(n_fold)

        

        #split z and X

        z_train = train_data[:,-1]

        xy_train = train_data[:,0:-1]

        z_test = test_data[:,-1]

        xy_test = test_data[:,0:-1]

        

        # Fit training data

        X_train = make_X_matrix_new(xy_train.T[0],xy_train.T[1],poly+1)

        #A = np.transpose(X_train) @ X_train

        #A = X_train.T.dot(X_train)

        #X_XT=SVDinv(A)

        

        #np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y

        beta = calc_beta(X_train, z_train)

        #beta=np.transpose(X_train).dot(np.linalg.inv(X_XT)).dot(z_train)

        

        #n, p = X_train.shape

        #sigma_inv = np.zeros((p, n))



        #u, sigma, vt = np.linalg.svd(X_train)

        #sigma_inv[:min(n, p), :min(n, p)] = np.diag(1 / sigma)

        #beta = (vt.T @ sigma_inv @ u.T) @ z_train

        

        # Do prediction on test and train data

        z_pred_test=predict_new(xy_test.T[0],xy_test.T[1],poly+1,beta)

        z_pred_train=predict_new(xy_train.T[0],xy_train.T[1],poly+1,beta)

        pred_test[:,i]=predict_new(xy_test.T[0],xy_test.T[1],poly+1,beta)

        pred_train[:,i]=predict_new(xy_train.T[0],xy_train.T[1],poly+1,beta)

        

        # Append results to arrays and lists

        r2_.append(r2_score(z_test,z_pred_test))

        

    train_error[poly] = np.mean( np.mean((z_train.reshape(z_train.shape[0],1) - pred_train)**2, axis=1, keepdims=True) )   

    error[poly] = np.mean( np.mean((z_test.reshape(z_test.shape[0],1) - pred_test)**2, axis=1, keepdims=True) )

    bias[poly] = np.mean( (z_test.reshape(z_test.shape[0],1) - np.mean(pred_test, axis=1, keepdims=True))**2 )

    variance[poly] = np.mean( np.var(pred_test, axis=1, keepdims=True) )



    print('Polynomial degree:', poly+1)

    print('Error:', error[poly])

    print('Bias^2:', bias[poly])

    print('Var:', variance[poly])

    print('{} >= {} + {} = {}'.format(error[poly], bias[poly], variance[poly], bias[poly]+variance[poly]))

        

    #plotting prediction based on all data 

    

    z_pred_for_plot = predict_new(x_and_y_scaled.T[0],x_and_y_scaled.T[1],poly+1,beta)

    fig = plt.figure(figsize=(32,12))

    ax = fig.gca(projection ='3d')

    surf = ax.plot_surface(x,y,z_pred_for_plot.reshape(n,n),cmap=cm.coolwarm, linewidth = 0, antialiased=False)

    ax.set_zlim(-0.10,1.40)

    ax.zaxis.set_major_locator(LinearLocator(10))

    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf,shrink=0.5, aspect=5)

    fig.suptitle("A {} degree polynomial fit of Franke function using OLS and K-fold crossval".format(poly+1) ,fontsize="40", color = "black")

    fig.show()

        

    all_r2_ols_cv.append(r2_)

    mean_r2_ols_cv.append(np.mean(r2_))
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

#fig.suptitle("A {} degree polynomial fit of ECG data using Ridge with lambda {}".format(int(np.where(mean_r2 == np.amax(mean_r2))[0])+1,LAMBDA[int(np.where(mean_r2 == np.amax(mean_r2))[1])]) ,fontsize="40", color = "black")

#fig.savefig("Franke_function_{}deg_reg.png".format(degree))

fig.show()
n = 40

m = 40

x,y = xy_data(n)

z = FrankeFunction(x,y)



pol=10



X=make_X_matrix_new(x,y,pol)

theta = np.random.randn(X.shape[1],1)





eta = 0.001

Niterations = 10000





for iter in range(Niterations):

    gradients = 2.0/m*X.T @ ((X @ theta)-z.ravel().reshape(z.ravel().shape[0],1))

    theta -= eta*gradients

print("theta from own gd")

print(theta)
zpredict_ols = X.dot(theta)
print("MSE:",mean_squared_error(z.ravel(),zpredict_ols))

print("R2-score:",r2_score(z.ravel(),zpredict_ols))

fig = plt.figure(figsize=(32,12))

ax = fig.gca(projection ='3d')

surf = ax.plot_surface(x,y,zpredict_ols.reshape(x.shape[0],y.shape[1]),cmap=cm.coolwarm, linewidth = 0, antialiased=False)

ax.set_zlim(-0.10,1.40)

ax.zaxis.set_major_locator(LinearLocator(10))

ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf,shrink=0.5, aspect=5)

fig.suptitle("A {} degree polynomial fit of Franke function using OLS and K-fold crossval".format(pol) ,fontsize="40", color = "black")

fig.show()
def calc_beta_ridge(X,y, alpha):

    beta=np.linalg.inv(X.T.dot(X)+alpha * np.identity(X.shape[1])).dot(X.T).dot(y)

    return beta
n = 40

m = 60

x,y = xy_data(n)

z = FrankeFunction(x,y)



pol=20

lmbda = 0.0001



X=make_X_matrix_new(x,y,pol)

theta = np.random.randn(X.shape[1],1)





eta = 0.001

Niterations = 10000





for iter in range(Niterations):

    gradients = 2.0/m*X.T @ (X @ (theta)-z.ravel().reshape(z.ravel().shape[0],1))+2*lmbda*theta

    theta -= eta*gradients

print("theta from own gd")

print(theta)
zpredict_ridge = X.dot(theta)
print("MSE:",mean_squared_error(z.ravel(),zpredict_ridge))

print("R2-score:",r2_score(z.ravel(),zpredict_ridge))

fig = plt.figure(figsize=(32,12))

ax = fig.gca(projection ='3d')

surf = ax.plot_surface(x,y,zpredict_ridge.reshape(x.shape[0],y.shape[1]),cmap=cm.coolwarm, linewidth = 0, antialiased=False)

ax.set_zlim(-0.10,1.40)

ax.zaxis.set_major_locator(LinearLocator(10))

ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf,shrink=0.5, aspect=5)

fig.suptitle("A {} degree polynomial fit of Franke function using OLS and K-fold crossval".format(pol) ,fontsize="40", color = "black")

fig.show()
z = np.mean(norm_data[:100],axis=0)

x = np.arange(500)

y = np.arange(12)

x,y=np.meshgrid(x,y)



m = 10

pol=1



X=make_X_matrix_new(x,y,pol)

theta = np.random.randn(X.shape[1],1)





eta = 0.000000001

Niterations = 100





for iter in range(Niterations):

    gradients = 2.0/m*X.T @ ((X @ theta)-z.ravel().reshape(z.ravel().shape[0],1))

    theta -= eta*gradients

print("theta from own gd")

print(theta)
zpredict_ols_ecg = X.dot(theta)
print("MSE:",mean_squared_error(z.ravel(),zpredict_ols_ecg))

print("R2-score:",r2_score(z.ravel(),zpredict_ols_ecg))

fig = plt.figure(figsize=(32,12))

ax = fig.gca(projection ='3d')

surf = ax.plot_surface(x,y,zpredict_ols_ecg.reshape(x.shape[0],y.shape[1]),cmap=cm.bone, linewidth = 0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))

ax.view_init(20, 75)

ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf,shrink=0.5, aspect=5)

#fig.suptitle("A {} degree polynomial fit of ECG data using Ridge with lambda {}".format(int(np.where(mean_r2 == np.amax(mean_r2))[0])+1,LAMBDA[int(np.where(mean_r2 == np.amax(mean_r2))[1])]) ,fontsize="40", color = "black")

#fig.savefig("Franke_function_{}deg_reg.png".format(degree))

fig.show()
z = np.mean(norm_data[:100],axis=0)

x = np.arange(500)

y = np.arange(12)

x,y=np.meshgrid(x,y)



m = 60



pol=1

lmbda = 0.0001



X=make_X_matrix_new(x,y,pol)

theta = np.random.randn(X.shape[1],1)





eta = 0.0000000001

Niterations = 10





for iter in range(Niterations):

    gradients = 2.0/m*X.T @ (X @ (theta)-z.ravel().reshape(z.ravel().shape[0],1))+2*lmbda*theta

    theta -= eta*gradients

print("theta from own gd")

print(theta)
zpredict_ridge_ecg = X.dot(theta)
print("MSE:",mean_squared_error(z.ravel(),zpredict_ridge_ecg))

print("R2-score:",r2_score(z.ravel(),zpredict_ridge_ecg))

fig = plt.figure(figsize=(32,12))

ax = fig.gca(projection ='3d')

surf = ax.plot_surface(x,y,zpredict_ridge_ecg.reshape(x.shape[0],y.shape[1]),cmap=cm.bone, linewidth = 0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))

ax.view_init(20, 75)

ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf,shrink=0.5, aspect=5)

#fig.suptitle("A {} degree polynomial fit of ECG data using Ridge with lambda {}".format(int(np.where(mean_r2 == np.amax(mean_r2))[0])+1,LAMBDA[int(np.where(mean_r2 == np.amax(mean_r2))[1])]) ,fontsize="40", color = "black")

#fig.savefig("Franke_function_{}deg_reg.png".format(degree))

fig.show()
x,y = xy_data(n)

z = FrankeFunction(x,y)



x_and_y=np.hstack((x.ravel().reshape(x.ravel().shape[0],1),y.ravel().reshape(y.ravel().shape[0],1)))



scaler = StandardScaler()

scaler.fit(x_and_y)

x_and_y_scaled = scaler.transform(x_and_y)



from sklearn.linear_model import SGDRegressor

sgdreg = SGDRegressor(max_iter = 50, penalty=None, eta0=0.1)

sgdreg.fit(x_and_y_scaled,z.ravel())

print(sgdreg.intercept_, sgdreg.coef_)
SK_pred=sgdreg.predict(x_and_y_scaled)
print("MSE:",mean_squared_error(z.ravel(),SK_pred))

print("R2-score:",r2_score(z.ravel(),SK_pred))

fig = plt.figure(figsize=(32,12))

ax = fig.gca(projection ='3d')

surf = ax.plot_surface(x,y,SK_pred.reshape(x.shape[0],y.shape[1]),cmap=cm.coolwarm, linewidth = 0, antialiased=False)

ax.set_zlim(-0.10,1.40)

ax.zaxis.set_major_locator(LinearLocator(10))

ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf,shrink=0.5, aspect=5)

fig.suptitle("A {} degree polynomial fit of Franke function using OLS and K-fold crossval".format(pol) ,fontsize="40", color = "black")

fig.show()
n=40

x,y = xy_data(n)

z = FrankeFunction(x,y)



m = 1

n_iter = n**2

pol=10

n_epochs = 50

t0, t1 = 5, 50

def learning_schedule(t):

    return t0/(t+t1)



X=make_X_matrix_new(x,y,pol)

theta = np.random.randn(X.shape[1],1)
for epoch in range(n_epochs):

    for i in range(n_iter):

        random_index = np.random.randint(n_iter)

        xi = X[random_index:random_index+1]

        zi = z.ravel()[random_index:random_index+1]

        gradients = 2.0/m * xi.T @ ((xi @ theta)-zi)

        eta = learning_schedule(epoch*n_iter+i)

        theta = theta - eta*gradients

print("theta from own sdg")

print(theta)
ypredict = X.dot(theta)
print("MSE:",mean_squared_error(z.ravel(),ypredict))

print("R2-score:",r2_score(z.ravel(),ypredict))

fig = plt.figure(figsize=(32,12))

ax = fig.gca(projection ='3d')

surf = ax.plot_surface(x,y,ypredict.reshape(x.shape[0],y.shape[1]),cmap=cm.coolwarm, linewidth = 0, antialiased=False)

ax.set_zlim(-0.10,1.40)

ax.zaxis.set_major_locator(LinearLocator(10))

ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf,shrink=0.5, aspect=5)

fig.suptitle("A {} degree polynomial fit of Franke function using OLS and K-fold crossval".format(pol) ,fontsize="40", color = "black")

fig.show()
n=40

x,y = xy_data(n)

z = FrankeFunction(x,y)

#x_and_y=np.hstack((x.ravel().reshape(x.ravel().shape[0],1),y.ravel().reshape(y.ravel().shape[0],1)))



m = 1

n_iter = n**2

pol=10

lmbda = 0.001

n_epochs = 100

t0, t1 = 5, 50

def learning_schedule(t):

    return t0/(t+t1)



#scaler = StandardScaler()

#scaler.fit(x_and_y)

#x_and_y_scaled = scaler.transform(x_and_y)

X=make_X_matrix_new(x,y,pol)

theta = np.random.randn(X.shape[1],1)
for epoch in range(n_epochs):

    for i in range(n_iter):

        random_index = np.random.randint(n_iter)

        xi = X[random_index:random_index+1]

        zi = z.ravel()[random_index:random_index+1]

        gradients = 2.0/m * xi.T @ ((xi @ theta)-zi)+2*lmbda*theta

        eta = learning_schedule(epoch*n_iter+i)

        theta = theta - eta*gradients

print("theta from own sdg")

print(theta)
ypredict = X.dot(theta)
print("MSE:",mean_squared_error(z.ravel(),ypredict))

print("R2-score:",r2_score(z.ravel(),ypredict))

fig = plt.figure(figsize=(32,12))

ax = fig.gca(projection ='3d')

surf = ax.plot_surface(x,y,ypredict.reshape(x.shape[0],y.shape[1]),cmap=cm.coolwarm, linewidth = 0, antialiased=False)

ax.set_zlim(-0.10,1.40)

ax.zaxis.set_major_locator(LinearLocator(10))

ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf,shrink=0.5, aspect=5)

fig.suptitle("A {} degree polynomial fit of Franke function using OLS and K-fold crossval".format(pol) ,fontsize="40", color = "black")

fig.show()
z = np.mean(norm_data[:100],axis=0)

x = np.arange(500)

y = np.arange(12)

x,y=np.meshgrid(x,y)





m = 1

n_iter = 10

pol=3

n_epochs = 10

t0, t1 = 0.00001, 50

def learning_schedule(t):

    return t0/(t+t1)



X=make_X_matrix_new(x,y,pol)

theta = np.random.randn(X.shape[1],1)
for epoch in range(n_epochs):

    for i in range(n_iter):

        random_index = np.random.randint(n_iter)

        xi = X[random_index:random_index+1]

        zi = z.ravel()[random_index:random_index+1]

        gradients = 2.0/m * xi.T @ ((xi @ theta)-zi)

        eta = learning_schedule(epoch*n_iter+i)

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

#fig.suptitle("A {} degree polynomial fit of ECG data using Ridge with lambda {}".format(int(np.where(mean_r2 == np.amax(mean_r2))[0])+1,LAMBDA[int(np.where(mean_r2 == np.amax(mean_r2))[1])]) ,fontsize="40", color = "black")

#fig.savefig("Franke_function_{}deg_reg.png".format(degree))

fig.show()
z = np.mean(norm_data[:100],axis=0)

x = np.arange(500)

y = np.arange(12)

x,y=np.meshgrid(x,y)



m = 1

n_iter =10

pol=3

lmbda = 0.000000001

n_epochs = 50

t0, t1 = 0.0001, 50

def learning_schedule(t):

    return t0/(t+t1)



X=make_X_matrix_new(x,y,pol)

theta = np.random.randn(X.shape[1],1)
for epoch in range(n_epochs):

    for i in range(n_iter):

        random_index = np.random.randint(n_iter)

        xi = X[random_index:random_index+1]

        zi = z.ravel()[random_index:random_index+1]

        gradients = 2.0/m * xi.T @ ((xi @ theta)-zi)+2*lmbda*theta

        eta = learning_schedule(epoch*n_iter+i)

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

#fig.suptitle("A {} degree polynomial fit of ECG data using Ridge with lambda {}".format(int(np.where(mean_r2 == np.amax(mean_r2))[0])+1,LAMBDA[int(np.where(mean_r2 == np.amax(mean_r2))[1])]) ,fontsize="40", color = "black")

#fig.savefig("Franke_function_{}deg_reg.png".format(degree))

fig.show()
z = np.mean(norm_data[:100],axis=0)

x = np.arange(500)

y = np.arange(12)

x,y=np.meshgrid(x,y)

x_and_y=np.hstack((x.ravel().reshape(x.ravel().shape[0],1),y.ravel().reshape(y.ravel().shape[0],1)))



m = 1

n_iter =1000

pol=8

lmbda = 0.001

n_epochs = 10

t0, t1 = 0.001, 50

def learning_schedule(t):

    return t0/(t+t1)



scaler = StandardScaler()

scaler.fit(x_and_y)

x_and_y_scaled = scaler.transform(x_and_y)



X=make_X_matrix_new(x_and_y_scaled.T[0],x_and_y_scaled.T[1],pol)

theta = np.random.randn(X.shape[1],1)
for epoch in range(n_epochs):

    for i in range(n_iter):

        random_index = np.random.randint(n_iter)

        xi = X[random_index:random_index+1]

        zi = z.ravel()[random_index:random_index+1]

        gradients = 2.0/m * xi.T @ ((xi @ theta)-zi)+2*lmbda*theta

        eta = learning_schedule(epoch*n_iter+i)

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

#fig.suptitle("A {} degree polynomial fit of ECG data using Ridge with lambda {}".format(int(np.where(mean_r2 == np.amax(mean_r2))[0])+1,LAMBDA[int(np.where(mean_r2 == np.amax(mean_r2))[1])]) ,fontsize="40", color = "black")

#fig.savefig("Franke_function_{}deg_reg.png".format(degree))

fig.show()