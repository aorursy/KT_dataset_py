import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
datafr = pd.read_csv('../input/smoker_data.csv')

datafr.info()

datafr['Smoker'] = datafr['Smoker'].astype(int)
datafr.head()
X = datafr.iloc[:,3:5].values

Y = datafr.iloc[:,5].values

rand_index = np.random.permutation(np.size(X,0)) # random shuffles

k = 5 # number of folds

train_test_ratio = (k-1)/k

train_index = np.zeros(shape=(k,int(train_test_ratio*np.size(X,0))))

test_index = np.zeros(shape=(k,int((1/k)*np.size(X,0))))
# define logistic regression solver 

def newtonGD(x,y,max_ite):

    weight_vec = np.zeros((3,1))

    for i in range(max_ite):

        h_x = 1/(1+np.exp(-x.dot(weight_vec)))

        #cost_val_newton[i] = -(1/(np.size(x,0)))*sum(y*np.log(h_x) + (1-y)*np.log(1-h_x))

        grad = (1/(np.size(x,0)))*(x.T.dot((h_x-y)))

        # hessian matrix

        H = (1/(np.size(x,0)))*(x.T.dot(np.diag(h_x.reshape(np.size(x,0),))).dot(np.diag((1-h_x).reshape(np.size(x,0),))).dot(x))

        weight_vec = weight_vec - np.linalg.pinv(H).dot(grad)

    return weight_vec
# define accuracy calculator

def misclass(y,y_predict):

    mis=0;indexmis=[]

    num_samples = np.size(y_predict,0)

    for i in range(num_samples):

        if y[i]== 1 and y_predict[i]<=0.5:

            mis = mis + 1

            indexmis.append(i)

        elif y[i]== 0 and y_predict[i]>0.5:

            mis = mis + 1

            indexmis.append(i)

    print('Number of misclassified data = ' + str(mis))

    accuracy = 100*(num_samples - mis)/num_samples

    print('Accuracy of classifier = ' + str(accuracy))

    return indexmis,accuracy 
accuracy_all = np.zeros((k,1))   

for ii in range(k):

    #ii = 0

    jj = ii + 1

    test_index[ii,:] = rand_index[ii*int(np.size(X,0)/k):jj*int(np.size(X,0)/k)]

    train_index[ii,:] = np.setdiff1d(rand_index,test_index[ii,:])

    ### Train model    

    Y_in_train = Y[train_index[ii,:].astype(int)].reshape(int(train_test_ratio*np.size(X,0)),1)

    X_in_train = X[train_index[ii,:].astype(int),:]

    

    X_in_train_pad = np.concatenate((np.ones((np.size(X_in_train,0),1)),X_in_train),axis=1)

    weight_vec = newtonGD(X_in_train_pad,Y_in_train,20)

    

    x_plot = np.linspace(min(X_in_train[:,0]),max(X_in_train[:,0]),20)

    y_plot = -weight_vec[0]/weight_vec[2] - (weight_vec[1]/weight_vec[2])*x_plot

    

    plt.scatter(X_in_train[:,0],X_in_train[:,1],c=Y[train_index[ii,:].astype(int)],label='train data')

    plt.plot(x_plot,y_plot,'r',label='Newtons method' )

    plt.xlabel('Diastolic');plt.ylabel('Systolic');plt.title('Train model with train data')

    plt.legend()

    plt.show()

    

    ### Test data used for testing model accuracy

    Y_in_test = Y[test_index[ii,:].astype(int)].reshape(int((1/k)*np.size(X,0)),1)

    X_in_test = X[test_index[ii,:].astype(int),:]

    

    X_in_test_pad = np.concatenate((np.ones((np.size(X_in_test,0),1)),X_in_test),axis=1)

    Y_predict_test = 1/(1 + np.exp(-X_in_test_pad.dot(weight_vec)))

    

    ind_mix,accuracy = misclass(Y_in_test,Y_predict_test)

    accuracy_all[ii] = accuracy

    plt.figure(figsize=(10,5))

    plt.scatter(X_in_test[:,0],X_in_test[:,1],c=Y[test_index[ii,:].astype(int)],label='test data')

    plt.scatter(X_in_test[ind_mix,0],X_in_test[ind_mix,1],marker='s',facecolors='none',edgecolors='k',s=80,label='misclassified data')

    plt.plot(x_plot,y_plot,'r',label='Newtons method' )

    plt.xlabel('Diastolic');plt.ylabel('Systolic');plt.title('Test model performance with test data ')

    plt.legend()

    plt.show()
#Last, calculate average model accuracy from 5 different runs

print('Average accuracy for ' +str(k) + '-fold cross validations is: ' +str(np.mean(accuracy_all)))
