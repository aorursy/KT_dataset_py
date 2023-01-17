import numpy as np

import matplotlib.pyplot as plt 

import pandas as pd 

import seaborn as sns 

#Calibration 

n=100

alpha=0

beta=1

gamma=np.array([alpha,beta])



M=1000 # No. of simulations 
#generatin the S

M=1000



#Experiment 1 

q=[0.10,0.50,0.90]



ones=np.ones(n)



#Generating the weights --- i.e. gamma hat 



gamma_hat=np.empty((1000,2,3))

for j in range(len(q)):

    for i in range(M):

        s=(2*np.random.binomial(1,q[j],size=n) -1).reshape(-1,1)

        u=np.random.rand(n,1)

        X=np.column_stack((ones.reshape(-1,1),np.power(np.multiply(s,u),2)))

        Y=X@gamma.reshape(-1,1) +np.random.normal(0,1,n).reshape(-1,1)

        gamma_hat[i,:,j] =((np.linalg.inv(X.T@X))@(X.T@Y)).T





# The matrix of predictions -- i.e. y_hat -- for the first experiment 



Y_hat_11=gamma_hat[:,:,0]@np.array([1,-0.5])

Y_hat_12=gamma_hat[:,:,0]@np.array([1,0])

Y_hat_13=gamma_hat[:,:,0]@np.array([1,0.5])



#Estimation errors 

#The CEF is basically the true beta into true X:

CEF_11=gamma@np.array([1,-0.5**2])

CEF_12=gamma@np.array([1,0])

CEF_13=gamma@np.array([1,0.5**2])



err_11=Y_hat_11-CEF_11

err_12=Y_hat_12-CEF_12

err_13=Y_hat_13-CEF_13

# The matrix of predictions -- i.e. y_hat -- for the second experiment 



Y_hat_21=gamma_hat[:,:,1]@np.array([1,-0.5])

Y_hat_22=gamma_hat[:,:,1]@np.array([1,0])

Y_hat_23=gamma_hat[:,:,1]@np.array([1,0.5])



#Estimation errors 

#The CEF is basically the true beta into true X:

CEF_21=gamma@np.array([1,-0.5**2])

CEF_22=gamma@np.array([1,0])

CEF_23=gamma@np.array([1,0.5**2])



err_21=Y_hat_21-CEF_21

err_22=Y_hat_22-CEF_22

err_23=Y_hat_23-CEF_23

# The matrix of predictions -- i.e. y_hat -- for the third experiment 



Y_hat_31=gamma_hat[:,:,2]@np.array([1,-0.5])

Y_hat_32=gamma_hat[:,:,2]@np.array([1,0])

Y_hat_33=gamma_hat[:,:,2]@np.array([1,0.5])



#Estimation errors 

#The CEF is basically the true beta into true X:

CEF_31=gamma@np.array([1,-0.5**2])

CEF_32=gamma@np.array([1,0])

CEF_33=gamma@np.array([1,0.5**2])



err_31=Y_hat_31-CEF_31

err_32=Y_hat_32-CEF_32

err_33=Y_hat_33-CEF_33

#Just add the plots 

n_bins = 20



fig, ax = plt.subplots(3, 3, sharey=True, tight_layout=True)



# First experiment i.e. q=0.1

ax[0,0].hist(err_11, bins=n_bins)

ax[0,1].hist(err_12, bins=n_bins)

ax[0,2].hist(err_13,bins=n_bins)



#Second Experiment i.e. q=0.5

ax[1,0].hist(err_21, bins=n_bins)

ax[1,1].hist(err_22, bins=n_bins)

ax[1,2].hist(err_23,bins=n_bins)



#Third Experiment i.e. q=0.9

ax[2,0].hist(err_31, bins=n_bins)

ax[2,1].hist(err_32, bins=n_bins)

ax[2,2].hist(err_33,bins=n_bins)
