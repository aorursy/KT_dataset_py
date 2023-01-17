#Importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
#Loading the dataset

data = pd.read_csv('../input/creditcard.csv')

data.head()
#Visualizing the target variable

target = pd.value_counts(data['Class'],sort=True).sort_index()

target.plot(kind = 'bar')

plt.title("Fraud counts")

plt.xlabel("class")

plt.ylabel("Frequency")

#Normalising the amount column

from sklearn.preprocessing import StandardScaler

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))

data = data.drop(['Time','Amount'],axis=1)

data.head()
X = data.ix[:,data.columns !='Class']

y = data.ix[:,data.columns =='Class']
#Number of data points in minority class

number_fraud_records = len(data[data.Class == 1])

fraud_indices = np.array(data[data.Class == 1].index)
#Indices of normal Class

normal_indices = data[data.Class == 0].index
#Randomly selecting x numbers

random_normal_indices = np.random.choice(normal_indices,number_fraud_records,replace = False)

random_normal_indices = np.array(random_normal_indices)
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
under_sample_data = data.iloc[under_sample_indices,:]
X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']

y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']
#showing ratio

print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))

print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))

print("Total number of transactions in resampled data: ", len(under_sample_data))
from sklearn.cross_validation import train_test_split



# Whole dataset

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)



print("Number transactions train dataset: ", len(X_train))

print("Number transactions test dataset: ", len(X_test))

print("Total number of transactions: ", len(X_train)+len(X_test))



X_train_undersample,X_test_undersample,y_train_undersample,y_test_undersample = train_test_split(X_undersample,y_undersample,test_size = 0.3, random_state = 0)



print("")

print("Number of transactions train dataset: ", len(X_train_undersample))

print("Number of transactions test dataset: ", len(X_test_undersample))

print("Total number of transactions: ", len(X_train_undersample)+len(X_test_undersample))
#Logistic Regression classifier



from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 

from sklearn.cross_validation import KFold, cross_val_score
def Kfold_scores(x_train_data,y_train_data):

    fold = KFold(len(y_train_data),5,shuffle = False)

    C_param_range = [0.01,0.1,1,10,100]

    results_table = pd.DataFrame(index = range(len(C_param_range),2), columns = ['C_parameter','Mean recall score'])

    results_table['C_parameter'] = C_param_range

    j = 0

    for C_param in C_param_range:

        print('-------------------------')

        print('C parameter: ', C_param)

        print('-------------------------')

        print('')

        recall_accs = []

        for iteration, indices in enumerate(fold,start=1):

            lr = LogisticRegression(C = C_param, penalty = 'l1')

#Predicting using train and test data

            lr.fit(x_train_data.iloc[indices[0],:],y_train_data.iloc[indices[0],:].values.ravel())

            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)

            recall_acc = recall_score(y_train_data.iloc[indices[1],:].values,y_pred_undersample)

            recall_accs.append(recall_acc)

            print('Iteration ', iteration,': recall score = ', recall_acc)

        # mean value of those recall scores

        results_table.ix[j,'Mean recall score'] = np.mean(recall_accs)

        j += 1

        print('')

        print('Mean recall score ', np.mean(recall_accs))

        print('')

    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']

    print("Best C score:",best_c)

    return best_c

best_c = Kfold_scores(X_train_undersample,y_train_undersample)