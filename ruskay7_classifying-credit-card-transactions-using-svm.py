#importing the libraries

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV

data= pd.read_csv("../input/creditcard.csv")

print(data.head())
count_Class=pd.value_counts(data["Class"], sort= True)

count_Class.plot(kind= 'bar')
No_of_frauds= len(data[data["Class"]==1])

No_of_normals = len(data[data["Class"]==0])

print("The number of fraudulent transactions( Class 1) are: ", No_of_frauds)

print("The number of normal transactions( Class 0) are: ", No_of_normals)

total= No_of_frauds + No_of_normals

Fraud_percent= (No_of_frauds / total)*100

Normal_percent= (No_of_normals / total)*100

print("Class 0 percentage = ", Normal_percent)

print("Class 1 percentage = ", Fraud_percent)
#list of fraud indices

fraud_index= np.array(data[data["Class"]==1].index)



#getting the list of normal indices from the full dataset

normal_index= data[data["Class"]==0].index



#choosing random normal indices equal to the number of fraudulent transactions

random_normal_indices= np.random.choice(normal_index, No_of_frauds, replace= False)

random_normal_indices= np.array(random_normal_indices)



# concatenate fraud index and normal index to create a list of indices

undersampled_indices= np.concatenate([fraud_index, random_normal_indices])



#use the undersampled indices to build the undersampled_data dataframe

undersampled_data= data.iloc[undersampled_indices, :]



print(undersampled_data.head())
No_of_frauds_sampled= len(undersampled_data[undersampled_data["Class"]== 1])



No_of_normals_sampled = len(undersampled_data[undersampled_data["Class"]== 0])



print("The number of fraudulent transactions( Class 1) are: ", No_of_frauds_sampled)

print("The number of normal transactions( Class 0) are: ", No_of_normals_sampled)

total_sampled= No_of_frauds_sampled + No_of_normals_sampled

print("The total number of rows of both classes are: ", total_sampled)



Fraud_percent_sampled= (No_of_frauds_sampled / total_sampled)*100

Normal_percent_sampled= (No_of_normals_sampled / total_sampled)*100

print("Class 0 percentage = ", Normal_percent_sampled)

print("Class 1 percentage = ", Fraud_percent_sampled)



#Check the data count now

count_sampled=pd.value_counts(undersampled_data["Class"], sort= True)

count_sampled.plot(kind= 'bar')
#We have to scale the Amount feature before fitting our model to our dataset



sc= StandardScaler()

undersampled_data["scaled_Amount"]=  sc.fit_transform(undersampled_data.iloc[:,29].values.reshape(-1,1))



#dropping time and old amount column

undersampled_data= undersampled_data.drop(["Time","Amount"], axis= 1)



print(undersampled_data.head())
X= undersampled_data.iloc[:, undersampled_data.columns != "Class"].values



y= undersampled_data.iloc[:, undersampled_data.columns == "Class"].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 0)

print("The split of the under_sampled data is as follows")

print("X_train: ", len(X_train))

print("X_test: ", len(X_test))

print("y_train: ", len(y_train))

print("y_test: ", len(y_test))
#Using the gaussian kernel to build the initail model. Let us see if this is the best parameter later

classifier= SVC(C= 1, kernel= 'rbf', random_state= 0)

classifier.fit(X_train, y_train.ravel())

#Predict the class using X_test

y_pred = classifier.predict(X_test)
#cm1 is the confusion matrix 1 which uses the undersampled dataset

cm1 = confusion_matrix(y_test, y_pred)



def confusion_matrix_1(CM):

    fig, ax = plot_confusion_matrix(conf_mat=CM)

    plt.title("The Confusion Matrix 1 of Undersampled dataset")

    plt.ylabel("Actual")

    plt.xlabel("Predicted")

    plt.show()



    print("The accuracy is "+str((CM[1,1]+CM[0,0])/(CM[0,0] + CM[0,1]+CM[1,0] + CM[1,1])*100) + " %")

    print("The recall from the confusion matrix is "+ str(CM[1,1]/(CM[1,0] + CM[1,1])*100) +" %")

confusion_matrix_1(cm1)
#Applying 10 fold cross validation

accuracies = cross_val_score(estimator = classifier, X=X_train, y = y_train.ravel(), cv = 10)

mean_accuracy= accuracies.mean()*100

std_accuracy= accuracies.std()*100

print("The mean accuracy in %: ", accuracies.mean()*100)

print("The standard deviation in % ", accuracies.std()*100)

print("The accuracy of our model in % is betweeen {} and {}".format(mean_accuracy-std_accuracy, mean_accuracy+std_accuracy))
#applying gridsearchCV to our classifier

#Specifying the parameters in dictionaries to try out different parameters.

#The GridSearchCV will try all the parameters and give us the best parameters



parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},

              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]



grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train.ravel())

best_accuracy = grid_search.best_score_

print("The best accuracy using gridSearch is", best_accuracy)



best_parameters = grid_search.best_params_

print("The best parameters for using this model is", best_parameters)
#fitting the model with the best parameters

classifier_with_best_parameters =  SVC(C= best_parameters["C"], kernel= best_parameters["kernel"], random_state= 0)

classifier_with_best_parameters.fit(X_train, y_train.ravel())

#predicting the Class 

y_pred_best_parameters = classifier_with_best_parameters.predict(X_test)

#creating a confusion matrix

#cm2 is the confusion matrix  which uses the best parameters

cm2 = confusion_matrix(y_test, y_pred_best_parameters)

#visualizing the confusion matrix

def confusion_matrix_2(CM):

    fig, ax = plot_confusion_matrix(conf_mat= CM)

    plt.title("The Confusion Matrix 2 of Undersampled dataset using best_parameters")

    plt.ylabel("Actual")

    plt.xlabel("Predicted")

    plt.show()

    print("The accuracy is "+str((CM[1,1]+CM[0,0])/(CM[0,0] + CM[0,1]+CM[1,0] + CM[1,1])*100) + " %")

    print("The recall from the confusion matrix is "+ str(CM[1,1]/(CM[1,0] + CM[1,1])*100) + " %")

confusion_matrix_2(cm2)

#also printing the confusion matrix 1 for comparison

confusion_matrix_1(cm1)

#creating a new dataset to test our model

datanew= data.copy()



#Now to test the model with the whole dataset

datanew["scaled_Amount"]=  sc.fit_transform(datanew["Amount"].values.reshape(-1,1))



#dropping time and old amount column

datanew= datanew.drop(["Time","Amount"], axis= 1)



#separating the x and y variables to fit our model

X_full= datanew.iloc[:, undersampled_data.columns != "Class"].values



y_full= datanew.iloc[:, undersampled_data.columns == "Class"].values





#splitting the full dataset into training and test set

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full, test_size= 0.25, random_state= 0)



print("The split of the full dataset is as follows")

print("X_train_full: ", len(X_train_full))

print("X_test_full: ", len(X_test_full))

print("y_train_full: ", len(y_train_full))

print("y_test_full: ", len(y_test_full))
#predicting y_pred_full_dataset

y_pred_full_dataset= classifier_with_best_parameters.predict(X_test_full)



#confusion matrix usign y_test_full and ypred_full

cm3 = confusion_matrix(y_test_full, y_pred_full_dataset)
def confusion_matrix_3(CM):

    fig, ax = plot_confusion_matrix(conf_mat= CM)

    plt.title("The Confusion Matrix 3 of full dataset using best_parameters")

    plt.ylabel("Actual")

    plt.xlabel("Predicted")

    plt.show()

    print("The accuracy is "+str((CM[1,1]+CM[0,0])/(CM[0,0] + CM[0,1]+CM[1,0] + CM[1,1])*100) + " %")

    print("The recall from the confusion matrix is "+ str(CM[1,1]/(CM[1,0] + CM[1,1])*100) +" %")

confusion_matrix_3(cm3)



#printing confusion matrix 2 for comparison with training set results

confusion_matrix_2(cm2)