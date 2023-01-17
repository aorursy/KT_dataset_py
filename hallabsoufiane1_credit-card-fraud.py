import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv(os.path.join(dirname, filename))

data.isna().sum()
data.sample(10)
# We split the dataframe in two categories good and fraudulous dataframes

data_fraud = data[data['Class'] == 1] # Fraudulant transactions

data_fraud = data_fraud.drop(['Time','Amount'],axis=1)

data_ok = data[data['Class'] == 0] # Good transactions

data_ok = data_ok.drop(['Time','Amount'],axis=1)
bins = np.linspace(-20, 20, 100)

fig,ax1 = plt.subplots(nrows=7,ncols=4,figsize=(24,22))

k = 0

for i in range(7):

    for j in range(4):

        ax1[i,j].hist(data_fraud.iloc[:,k], bins, alpha=0.5,color='tab:red');

        ax1[i,j].set_ylabel('Fraudulant transactions', color='tab:red', fontsize=10)

        ax1[i,j].tick_params(axis='y', labelcolor='tab:red')

        ax1[i,j].set_title(data_fraud.columns[k])

        ax2 = ax1[i,j].twinx()

        ax2.hist(data_ok.iloc[:,k], bins, alpha=0.5,color='tab:blue');

        #if (j%4==0) :

        ax2.set_ylabel('Good transactions', color='tab:blue', fontsize=10)

        ax2.tick_params(axis='y', labelcolor='tab:blue')

        k+=1

        plt.tight_layout()
def fraud_detection(X,limit,threshold):

    '''This function takes as an input the dataframe containing the transactions in each row

    and for all the variables see if there is one of them is out of the range 

    set by [low_lim, high_lim]. The threshhold is another parameter of our classifier

    that allow us to see how many hits a trasactions need to have to decide that it is a 

    fraudulous transaction'''

    

    # Get the number of hits for each row if the dataframe 

    hits = np.sum((abs(X) >= limit).astype(int), axis = 1)

    # If the number of hits is above the specified threshold 

    # then the transaction if a fraud then the output is 1 (fraudulous transacation)

    # otherwise it is 0 (good transaction)

    fraud = (hits >=threshold).astype(int)

    return fraud





def perf_measure(y_actual, y_hat):

    '''This function is to measure the performance of the classifier by

    calculating the precision, recall and f1 '''

    

    TP = 0 # True positive

    FP = 0 # False positive

    TN = 0 # True negative

    FN = 0 # False negative



    for i in range(len(y_hat)): 

        if y_actual[i]==y_hat[i]==1:

           TP += 1

        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:

           FP += 1

        if y_actual[i]==y_hat[i]==0:

           TN += 1

        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:

           FN += 1



    precision = TP / (TP + FP)

    recall = TP / (TP + FN)

    f1 = 2*(recall * precision) / (recall + precision)



    return precision, recall, f1
# Mask for the good/fraudulous transactions to split the data frames into

# train/test dataframes. We gat 70% of the data to train the classifier

msk_ok = np.random.rand(len(data_ok)) < 0.7

msk_fraud = np.random.rand(len(data_fraud)) < 0.7



# Splitting the datasets

train_data_ok = data_ok[msk_ok]

test_data_ok = data_ok[~msk_ok]



train_data_fraud = data_fraud[msk_fraud]

test_data_fraud = data_fraud[~msk_fraud]



# Concatenate the two dataframes and then shuffle them

train_data = pd.concat([train_data_ok, train_data_fraud])

train_data = train_data.sample(frac=1).reset_index(drop=True)

test_data = pd.concat([test_data_ok, test_data_fraud]) 

test_data = test_data.sample(frac=1).reset_index(drop=True)





X_train = train_data.loc[:, train_data.columns != 'Class'].values

y_train = train_data['Class'].values



X_test = test_data.loc[:, test_data.columns != 'Class'].values

y_test = test_data['Class'].values
limit = 5; 

threshold = 10

y_pred_train = fraud_detection(X_train,limit,threshold)

precision, recall, f1 = perf_measure(y_train, y_pred_train)



print('Performance of the classifier using the training data:')

print('Precision : %f' % precision)

print('Recall : %f' %recall)

print('F1 score : %f' %f1)

print('-'*40)



y_pred_test = fraud_detection(X_test,limit,threshold)

precision, recall, f1 = perf_measure(y_test, y_pred_test)



print('Performance of the classifier using the test data:')

print('Precision : %f' %precision)

print('Recall : %f' %recall)

print('F1 score : %f' %f1)

print('-'*40)
def grid_search_func(param1,param2):

    '''This function helpa gets the precision and recall for different values

    of the limit and thereshold parameter to find the best setting that gives an 

    optimal value for the hyperparameters of the classifier'''

    

    t = 0

    perform_matrix = np.zeros((param1.shape[0]*param2.shape[0],5))

    for i in range(param1.shape[0]):

        for j in range(param2.shape[0]):

            y_pred = fraud_detection(X_train, param1[i], param2[j])

            precision, recall, f1 = perf_measure(y_train, y_pred)

            perform_matrix[t,:] = [param1[i], param2[j], precision, recall, f1]

            t+=1

    return perform_matrix
limit_vector = np.arange(3,6,0.5)

threshold_vector = np.arange(1,15,2)



perf_matrix = grid_search_func(limit_vector,threshold_vector)
fig = plt.figure(figsize=(10,8))



for i in range(len(limit_vector)):

    plt.plot(perf_matrix[np.where(perf_matrix[:,0] == limit_vector[i])][:,2],

                perf_matrix[np.where(perf_matrix[:,0] == limit_vector[i])][:,3], 

                label = 'limit=%f' %limit_vector[i]);



plt.xlabel('Recall');

plt.ylabel('Precision');

plt.legend();
fig = plt.figure(figsize=(10,8))



for i in range(len(limit_vector)):

    plt.plot(perf_matrix[np.where(perf_matrix[:,0] == limit_vector[i])][:,1],

                perf_matrix[np.where(perf_matrix[:,0] == limit_vector[i])][:,4], 

                label = 'limit=%f' %limit_vector[i]);



plt.xlabel('Threshold');

plt.ylabel('F1 score');

plt.legend();