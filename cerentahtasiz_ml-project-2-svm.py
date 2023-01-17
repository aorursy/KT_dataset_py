!pip install libsvm
!pip install svm
!pip install cvxopt
import scipy.io as sci

import numpy as np

import random 

from libsvm.svmutil import *

from cvxopt import matrix, solvers
X_Y=sci.loadmat('../input/datamat/data.mat')

X_train, y_train = X_Y['X'][:150,:], X_Y['Y'][:150,:].flatten()

X_test, y_test = X_Y['X'][150:,:], X_Y['Y'][150:,:].flatten()

del X_Y



# defines train problem 

train_prob  = svm_problem(y_train, X_train)



#defines parameters

param = svm_parameter()

param.kernel_type = 0 # linear kernel 

param.svm_type = 0 # C-SVC 

param.C = 1e7 # the value of C is chosen as high number to obtain hard margin SVM
# trains the model according to defined train problem and parameter

model = svm_train(train_prob,param)
# calculates train classification accuracy: 

p_labs, p_acc, p_vals = svm_predict(y_train, X_train, model)
# calculates test classification accuracy

p_labs, p_acc, p_vals = svm_predict(y_test, X_test, model)
# calculates test accuracies of the model for changing kernel type for a fixed C:

param = svm_parameter()

param.svm_type = 0 # C-SVC 

param.C = 4

tArray = np.arange(0,4) 



for tval in tArray:

    param.kernel_type = tval

    model = svm_train(train_prob,param)

    print('Test accuracy for kernel type '+str(tval))

    p_labs, p_acc, p_vals = svm_predict(y_test, X_test, model)

    print('')

    
# calculates test accuracies of the model for changing value of parameter C for a fixed kernel type:

param = svm_parameter()

param.kernel_type = 0 # linear kernel 

param.svm_type = 0    # C-SVC 



powerArray = np.arange(-2,16)

for power in powerArray:

    param.C = 2**float(power)

    model = svm_train(train_prob,param)

    print('Accuracy for C value 2^'+str(power))

    p_labs, p_acc, p_vals = svm_predict(y_test, X_test, model)

    print('')

    
# calculates the number of support vector for changing value of C 

param = svm_parameter()

param.kernel_type = 0 # linear kernel 

param.svm_type = 0 # C-SVC



for power in powerArray:

    param.C = 2**float(power)  

    model = svm_train(train_prob,param)

    print('Number of support vectors for C value 2^'+str(power))

    print(model.get_nr_sv())

    print('')
# trains model

param = svm_parameter()

param.kernel_type = 0 # linear kernel 

param.svm_type = 0 # C-SVC

param.C = 4

model = svm_train(train_prob,param)

# Excluding one of support vector and training.

sv_indice_list=np.array(model.get_sv_indices())-1

sv_indice=random.choice(sv_indice_list)  # indice zero starts with 1 in the get_sv_indices()

X_train_sv_exc=np.delete(X_train,sv_indice,axis=0)

y_train_sv_exc=np.delete(y_train,sv_indice,axis=0)

train_prob  = svm_problem(y_train_sv_exc,X_train_sv_exc)

model_sv_exc=svm_train(train_prob,param)



# Excluding one data point that is not a support vector and training

non_sv_indice=random.randint(1,X_train.shape[0])

while 1:

    non_sv_indice=random.randint(1,X_train.shape[0])

    if non_sv_indice not in sv_indice_list:

        break

X_train_non_sv_exc=np.delete(X_train,non_sv_indice,axis=0)

y_train_non_sv_exc=np.delete(y_train,non_sv_indice,axis=0)

train_prob=svm_problem(y_train_non_sv_exc,X_train_non_sv_exc)

model_non_sv_exc=svm_train(train_prob,param)
def weight(model_weight):

    SV=np.array(model_weight.get_SV())



    # Support vectors

    SV_matrix=np.zeros((len(SV),X_train.shape[1]))



    for i in range (0,SV_matrix.shape[0]):

        for j in range (1,SV_matrix.shape[1]+1):

            # In the SV dictionary, zero elements are not included. So, if any dictionary element is missing fill it with zero.

            try :

                SV_matrix[i,j-1]=SV[i][j]

            except:

                SV_matrix[i,j-1]=0



    # Support vector coefficient



    sv_coeff=np.array(model_weight.get_sv_coef())



    # weight calculation

    return np.dot(SV_matrix.T,sv_coeff)
# weight comparisons

print(np.linalg.norm(weight(model)-weight(model_sv_exc)))

print(np.linalg.norm(weight(model)-weight(model_non_sv_exc)))
# defines data points:

X = np.array([[0,0],[2,2],[2,0],[3,0]])



# defines labels:

y = np.array([[-1],[-1],[1],[1]])



# number of features:

d = len(X[0]) 
# determines coefficients for QP solver QP(P,q,G,h) where P = Q, q = p, G = A, h = c

Q = np.zeros((d+1,d+1))

Q[1:,1:]=np.eye(d)



p = np.zeros((d+1,1))



A = np.column_stack([y,np.multiply(y,X)])*-1. # multiplied with -1 to reverse inequality sign 



c = np.ones((len(X),1))*-1. # multiplied with -1 to reverse inequality sign 
# transforms numpy array into matrix form of CVXOPT library

Q = matrix(Q,tc='d')

p = matrix(p,tc='d')

A = matrix(A,tc='d')

c = matrix(c,tc='d')
# calculates optimum bias and weights 

u =solvers.qp(Q,p, A, c)['x']

print('')

b = np.array(u[0])

w = np.array(u[1:])
print('Bias Term:')

print(b)

print('Weights:')

print(w)