# Import the libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Import the dataset

data = pd.read_csv("../input/creditcard.csv")

data.head()
# PCA yields the directions (principal components) that maximize the variance of the data

# V1-V28 are from PCA processing, should be uncorrelated



# Q1: how to get the PCA analysis? feature selection ??  



#Plotting a heatmap to visualize the correlation between the variables

sns.heatmap(data.corr())
# As mentioned in the project, the data is imblaslanced, we can check the class distributions

sns.countplot("Class", data=data)
# for all feature, only amount is not scaled, so we can take a look at the distribution

# maybe the amount distribution is quite different between fraud vs non-fraud transaction

fraud_transacation = data[data["Class"]==1]

non_fraud_transacation= data[data["Class"]==0]

plt.figure(figsize=(10,6))

plt.subplot(121)

fraud_transacation.Amount.plot.hist(title="Fraud Transacation")

plt.subplot(122)

non_fraud_transacation.Amount.plot.hist(title="Non-Fraud Transaction")

# after the plot, we can see there is no clear difference between the two classes
# above plots show that most transactions are below 2.5k amount, so we can focus on the region

fraud_transacation = data[data["Class"]==1]

non_fraud_transacation= data[data["Class"]==0]

plt.figure(figsize=(10,6))

plt.subplot(121)

fraud_transacation[fraud_transacation["Amount"] <= 2500].Amount.plot.hist(title="Fraud Transacation")

plt.subplot(122)

non_fraud_transacation[non_fraud_transacation["Amount"] <= 2500].Amount.plot.hist(title="Non-Fraud Transaction")
from sklearn.preprocessing import StandardScaler



data["Normalized Amount"] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

data.drop(["Time","Amount"],axis=1,inplace=True)

data.head()
X = data.iloc[:, data.columns != 'Class'].values

y = data.iloc[:, data.columns == 'Class'].values
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)



print("Original number transactions train dataset: ", len(X_train))

print("Original number transactions test dataset: ", len(X_test))

print("Total number of transactions: ", len(X_train)+len(X_test))
# Number of data points in the minority class

number_records_fraud = len(data[data.Class == 1])

fraud_indices = np.array(data[data.Class == 1].index)



# Picking the indices of the normal classes

normal_indices = data[data.Class == 0].index



# Out of the indices we picked, randomly select "x" number (number_records_fraud)

random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)

random_normal_indices = np.array(random_normal_indices)



# Appending the 2 indices

under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])



# Under sample dataset

under_sample_data = data.iloc[under_sample_indices,:]



X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']

y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']



# Showing ratio

print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))

print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))

print("Total number of transactions in resampled data: ", len(under_sample_data))



# Undersampled dataset

X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample

                                                                                                   ,y_undersample

                                                                                                   ,test_size = 0.3

                                                                                                   ,random_state = 0)

print("")

print("Number transactions train dataset: ", len(X_train_undersample))

print("Number transactions test dataset: ", len(X_test_undersample))

print("Total number of transactions: ", len(X_train_undersample)+len(X_test_undersample))
from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import KFold, cross_val_score

from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
def printing_Kfold_scores(x_train_data,y_train_data):

    fold = KFold(len(y_train_data),5,shuffle=False) 



    # Different C parameters

    c_param_range = [0.01,0.1,1,10,100]



    results_table = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Mean recall score'])

    results_table['C_parameter'] = c_param_range



    # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]

    j = 0

    for c_param in c_param_range:

        print('-------------------------------------------')

        print('C parameter: ', c_param)

        print('-------------------------------------------')

        print('')



        recall_accs = []

        for iteration, indices in enumerate(fold,start=1):



            # Call the logistic regression model with a certain C parameter

            lr = LogisticRegression(C = c_param, penalty = 'l1')



            # Use the training data to fit the model. In this case, we use the portion of the fold to train the model

            # with indices[0]. We then predict on the portion assigned as the 'test cross validation' with indices[1]

            lr.fit(x_train_data.iloc[indices[0],:],y_train_data.iloc[indices[0],:].values.ravel())



            # Predict values using the test indices in the training data

            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)



            # Calculate the recall score and append it to a list for recall scores representing the current c_parameter

            recall_acc = recall_score(y_train_data.iloc[indices[1],:].values,y_pred_undersample)

            recall_accs.append(recall_acc)

            print('Iteration ', iteration,': recall score = ', recall_acc)



        # The mean value of those recall scores is the metric we want to save and get hold of.

        results_table.ix[j,'Mean recall score'] = np.mean(recall_accs)

        j += 1

        print('')

        print('Mean recall score ', np.mean(recall_accs))

        print('')



    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']

    

    # Finally, we can check which C parameter is the best amongst the chosen.

    print('*********************************************************************************')

    print('Best model to choose from cross validation is with C parameter = ', best_c)

    print('*********************************************************************************')

    

    return best_c
best_c = printing_Kfold_scores(X_train_undersample,y_train_undersample)
# Use this C_parameter to build the final model with the sampled training dataset and predict the classes in the test

# dataset

lr = LogisticRegression(C = best_c, penalty = 'l1') # l2 is about 90% recall

lr.fit(X_train_undersample,y_train_undersample.values.ravel())

y_pred_undersample = lr.predict(X_test_undersample.values)



# Compute and plot confusion matrix

cnf_matrix = confusion_matrix(y_test_undersample,y_pred_undersample)

print("the recall for this model is :",cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]))

fig= plt.figure(figsize=(6,3))# to plot the graph

print("TP",cnf_matrix[1,1,]) # no of fraud transaction which are predicted fraud

print("TN",cnf_matrix[0,0]) # no. of normal transaction which are predited normal

print("FP",cnf_matrix[0,1]) # no of normal transaction which are predicted fraud

print("FN",cnf_matrix[1,0]) # no of fraud Transaction which are predicted normal

sns.heatmap(cnf_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5)

plt.title("Confusion_matrix")

plt.xlabel("Predicted_class")

plt.ylabel("Real class")

plt.show()
# Use this C_parameter to build the model with the sampling dataset and predict the classes in the whole test dataset

lr = LogisticRegression(C = best_c, penalty = 'l1')

lr.fit(X_train_undersample,y_train_undersample.values.ravel())

y_pred = lr.predict(X_test)



# Compute and plot confusion matrix

cnf_matrix = confusion_matrix(y_test,y_pred)



print("the recall for this model is :",cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]))

fig= plt.figure(figsize=(6,3))# to plot the graph

print("TP",cnf_matrix[1,1,]) # no of fraud transaction which are predicted fraud

print("TN",cnf_matrix[0,0]) # no. of normal transaction which are predited normal

print("FP",cnf_matrix[0,1]) # no of normal transaction which are predicted fraud

print("FN",cnf_matrix[1,0]) # no of fraud Transaction which are predicted normal

sns.heatmap(cnf_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5)

plt.title("Confusion_matrix")

plt.xlabel("Predicted_class")

plt.ylabel("Real class")

plt.show()