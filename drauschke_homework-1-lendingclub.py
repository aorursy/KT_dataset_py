#Python 3 environment



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #import the graphing functions

import seaborn as sns #graphing functions

from sklearn.model_selection import train_test_split #import split function

from sklearn.ensemble import RandomForestClassifier #random forest classifier tool

from sklearn import metrics #analysis tools

from sklearn.metrics import confusion_matrix #confusion matrix





    #"as" defines the functions so you dont have to write them out all the time 



import os

print(os.listdir("../input"))
data = pd.read_csv("../input/loan.csv", low_memory = False) 

       #Load the data using pandas read csv function, low_memory is turned on by default and 

       #allows the file to be processed in chunks default
x = data[(data.loan_status == "Fully Paid") | (data.loan_status == "Default")] 

       #Only keep data for loans whose status is "Fully Paid" or "Default
x["target"] = (data.loan_status == "Fully Paid")

       #Transform loans that are paid into a binary target variable 

       #also note that in Python3 True/False are actually 0 and 1 so no need to transform
#ANSWER

print("Remaining Observations: ", x.shape[0]) #print the first line of shape command (i.e. the number of rows)

print()

print("Remaining Features:     ", x.shape[1]) #print the second line of the shape command (i.e. the number of columns)
###############

#ANALYSIS BACKUP



#Original dataset

print("Original Data Rows and Columns            =", data.shape)

       #show the shape of the original dataset 



#New dataset which only has defaluted and repaid loans

print("Defaulted & Repaid Data: Rows and Columns =", x.shape)

print()

       #show the shape of the transformed dataset x then space

    

print("Number of Loans Repaid:   ", len(x[x.loan_status == "Fully Paid"]))

print("Number of Loans Defaulted:", len(x[x.loan_status =="Default"]))

       #show the number of repaid loans and the number of defaulted loans

    

       #We see that the transformation leads us to have a smaller set 

       #of observations and we have one addtional column "target"....

       #Note that there are only 31 defaults in a dataset of over a 

       #million repaid loans!!!!-> CONCERN

       #Probably not very useful for default prediction.

       #To check that we didnt accedentially lose defaulted loans when removing 

       #current loans we can count the number of defaulted loans in the original dataset



print()

print("Number of Loans Defaulted Original Dataset:", len(data[data.loan_status =="Default"]))

       #Show the number of defaulted loans in the entire dataset

    

       #We can see that in the entire dataset of 2.26m loans there are only 31 recorded defaults

       #-> CONCERN
#ANSWER



#HISTOGRAM

plt.hist(x["loan_amnt"],bins = 100)

      #plot histogram loan amounts across one hundred bins

plt.title("Distribution of Loan Amounts")

      #name title

plt.xlabel("Loan Amounts in $")

      #Label x axis

plt.ylabel("Number #")

      #Label y axis

plt.grid(axis="y")

      #add grid only to y axis (default would add to both axis)



#Notice that the histogram shows that people apply 

#for round $ numbers for their loans i.e. 5k, 10k, 15k, 10k, 25k

    

    

#SUMMARY STATISTICS

print("Mean =    $", round(np.mean(x.loan_amnt),2))

      #find the mean of loan amount (use print so that many outputs from one block of code)

print("Median =  $", round(np.median(x.loan_amnt),2))

      #find the median of loan amount

print("Maximum = $", round(np.max(x.loan_amnt),2))

      #find the maximum of loan amount

print("Std =     $", round(np.std(x.loan_amnt),2))

      #find the standard deviation of loan amount then add space
#ANSWER



#mean

print("Mean Interest Rate by Maturity (%):")

print(round(x.groupby(by="term").int_rate.mean(),2))

print()

      #show the summary statistics of interest rate grouped by the loan maturities



#standard deviation

print("Standard Deviation of Interest Rate by Maturity (%):")

print(round(x.groupby(by="term").int_rate.std(),2))

print()

      #show the summary statistics of interest rate grouped by the loan maturities



#boxplot 

bplot = sns.boxplot(x= x["term"], y= data["int_rate"], fliersize=0) 

    #define boxplot using seaborn, fliersize=0 turnsoff 

    #highlighting of outliers

bplot.axes.set_title("Interest Rate by Maturity", fontsize=16) #set title

bplot.set_xlabel("Maturity", fontsize=14) #set x axis label

bplot.set_ylabel("Annual Interest Rate (%)", fontsize=14) #set y axis label

print(bplot)
###############

#ANALYSIS BACKUP



print(x.groupby(by="term").int_rate.describe())

      #show the summary statistics of interest rate grouped by the loan maturities
#ANSWER



#Grade with highest interest rate

print("Debt Grade with Highest Average Interest Rate:  ",x.groupby(by="grade").int_rate.mean().idxmax())



#Average interest rate of grade G

print("Average Interest Rate for Grade G:              ", round(x[x.grade == "G"].int_rate.mean(),2),"%")
###############

#ANALYSIS BACKUP



#average interest rate by grade

print("Average Interest Rate by Grade (%)",x.groupby(by="grade").int_rate.mean())

print()

     #check that grade G is actually the highest



#box-plot by sub-grade (out of curiosity)

x.boxplot(column="int_rate",by="sub_grade",showfliers=False, grid=False, rot=90) 

     #turn off outliers, gridlinse, rotate x-axis labels 90 degrees



#ANSWER



print("Highest Realized Yield by Debt Grade (%):")

print((x.groupby("grade")["total_pymnt"].sum() / x.groupby("grade")["funded_amnt"].sum()-1)*100)



     #Highest realized yield is 32.5% in grade F
###############

#ANALYSIS BACKUP



print("Defaulted Loans by Grade:", x.groupby(["grade","loan_status"]).size())

print()

     #since there are so few defaulted loans almost meaningless to remove 



print("Maximum Interest Rate (%) on Repaid Loans by Sub-Grade:")

print(x[x.loan_status == "Fully Paid"].groupby(by="grade").int_rate.max())

     #only count loans that are repaid and then show maximum interest rate displayed by grade

     #notice that the maximum interest rate for G is lower than the maximum realized yield 

     #suggesting that additional fees must have been charged
#ANSWER

print("Number of Applications by Type:", x.groupby("application_type").size())

      #Shows the number of each type of application

    

#It seems to be very unevenly distributed. With a larger dataset and 

#outside of a LendingClub environment this may be an interesting feature 

#because joint applicants give you recourse to two people (which may reduce 

#the possibility of default) but also may reduce the loss-given-default.
#ANSWER



model_dataset = x.loc[:, ["loan_amnt","funded_amnt","funded_amnt_inv","term","int_rate","emp_length","addr_state","verification_status","purpose","policy_code"]]

print("Model Dataset Shape:             ", model_dataset.shape)

          #create new dataset with the suggested variables - same row number but 10 columns



model_dataset2 = pd.get_dummies(data=model_dataset,columns=["addr_state","term","purpose","verification_status","policy_code", "emp_length"])

          #convert categorical data into dummies

    

feature_list = list(model_dataset2.columns)

          #saving list of features for later use



print("Dataset with Dummy Features Size:", model_dataset2.shape)

          #size of dataset with dummies. 

          #Note that the number of observations remails at 1,041,983

          #The number of columns is now at 86 this compares to 10 in the "model_dataset" dataset

          #This makes sense given that we have 6 dummy features with 81 unique entries

          #10+81-6+1(because "purpose" only has one entry) = 86
###############

#ANALYSIS BACKUP



#see how many unique terms there are for the features

print("States:", np.unique(x.addr_state))

print("  ")

          #51 unique entries

print("Term:", np.unique(x.term))

print("  ")

          #2 unique entries

print("Purpose:", np.unique(x.purpose))

print("  ")

          #14 unique entries

print("Verification Status:", np.unique(x.verification_status))

print("  ")

          #3 unique entries

print("Policy Code:", np.unique(x.policy_code))

print("  ")

          #this entry is 1 in all cases

          #employment length has 12 unique entries but does not print

          #all other listed features are numerical
#ANSWER



train_features, test_features, train_labels, test_labels = train_test_split(model_dataset2, x.target, test_size=0.67, train_size=0.33, random_state=42)

print("Training Features Shape:",train_features.shape)

print("Testing Features Shape:",test_features.shape)

print('Training Labels Shape:', train_labels.shape)

print('Testing Labels Shape:', test_labels.shape)

        #split data using a training size of 33% (test_size corresponts) and random state 42

        #notie the the lables (y) is taken from the original dataset "x" and the features comes from "model_dataset2"

        #we need to set random_state otherwise everytime the program runs we will get a different training set

        #print shape of training and test set.

        #Notice that the feature test set has 698,129 rows which corresponds to 67% of 1,041,983.

        
#ANSWER



#train model

rf = RandomForestClassifier(n_estimators = 100, max_depth = 4, random_state = 42)

       #we use the classifier becasue we have a binary output (otherwise we would use the regressor)

rf.fit(train_features,train_labels)

       #fit the model
predictions = rf.predict(test_features)

      #save the predictions for the test sample

      #note that the model rounds the estimated default probability to be either 0 or 1







print("Accuracy:",round(metrics.accuracy_score(test_labels, predictions)*100,4),"%")

       #accuracy of the predictions

       #accuracy of 99.9968%

       #the model is predicting that all applicants will repay and with such a low number of 

       #defaults we (only 22 defaults in an test sample of 698,129) a "dumb" approach 

       #of predicting that every single loan will be repaid yields an accuracy of 99.9968%

       #(1 - 22/698,129)
###############

#ANALYSIS BACKUP



#confirm that model is making "dumb" predictions

print("Types of Prediction:", np.unique(predictions))

print()

      #confirm that predictions are binary

      #Notice that the model is only making "True" predictions. This "dumb" 

      #approach will result in an illusionary accuracy ratio. We will have 

      #a 100% accuracy for repaid loans and a 0% accuracy for defaulted loans. 

      #Because the sample us unbalanced this will mean a very high accuracy 

      #ratio overall...



#see how many defaults there are in the test data

print("Number of Defaults in Test Data:", (len(test_labels)-test_labels.sum()))

print()

      #only 22 defaults

        

#Generate confusion matrix

def print_confusion_matrix(test_labels,predictions):

    cm = confusion_matrix(test_labels,predictions)

    print("Defaulting Applicant Rejected = ", cm[0][0])

    print("Defaulting Applicant Approved = ", cm[0][1])

    print("Paying Applicant Rejected =     ", cm[1][0])

    print("Paying Applicant Approved =     ", cm[1][1])

    #define the condusion matrix so that it prints output in easily interpretable way

    #note that the print settings call predefined parts of the output 2x2 matrix (i.e. true positive is 0;0 i.e. top-left)

    #to execute command print_confusion_matrix(test_default,predictions)

print_confusion_matrix(test_labels,predictions)



    #another way to see that all defaulting loan applicants would be approved by this model

    
predictions0 = np.ones(test_labels.shape)



print("Accuracy:",round(metrics.accuracy_score(test_labels, predictions0)*100,4),"%")
#ANSWER

#I adjusted for this imbalance in two ways: 

#(1) the nature of LendingClub's business model is such that they have no interest to hold 

#on to, or enforce, non-performing loans, so as soon as a loan is more than 120 days past due 

#they sell the loan to debt collectors. The loan is then part of the "charged-off" category. 

#See https://help.lendingclub.com/hc/en-us/articles/216127897-What-happens-when-a-loan-is-charged-off- 

#for details. Therefore, I included the charged-off category and renamed it "default" because that 

#is what this category is in reality



#(2) as a result of the above I had 261,686 defaults in a dataset of 1,303,638; to further account for 

#the imbalanced data I used "class_weight = "balanced"" in my random forest classifier to oversample from 

#the default group.
#Investigate and prepare data

print("Number by Status Types:", data.groupby("loan_status").size())

       #shows the number of loans in each category

x2 = data[(data.loan_status == "Fully Paid") | (data.loan_status == "Default") | (data.loan_status == "Charged Off")] 

print()

       #Only keep data for loans whose status is "Fully Paid" or "Default" or "Charged Off"

       #Keep "charged Off" bacause they are merely defaulted loans that LendingClub has sold to 

       #debt collectors (see answer to question 11 for details).

       #Also drop current or overdue loans to affect our predictions since a loan can turn into a

       #default tomorrow or an overdue loan can be repaid (thereby biasing our model). 

       #I dropped the "does not meet credit policy" items because there are not many 

       #of them and there is no clear description from LendingClub on what these items are.

    

x2 = x2.replace("Charged Off", "Default")

       #replace "charged off" with "default" to account for the fact that they are the same thing



print("Observations by Status Types:")

print(x2.groupby("loan_status").size())

print()

       #show shape of new dataset and the unique loan_status entries

       #we see that the new number of defaults is 261,686 which equals 261,655 + 31

        

x2["target2"] = (x2.loan_status == "Fully Paid")

       #Transform loans that are paid into a binary target variable 

    

print("New Dataset Shape:                ",x2.shape)

print()

       #check that the new target2 column has been added (146 vs 145)

    

model_data = x2.loc[:, ["loan_amnt","funded_amnt","funded_amnt_inv","term","int_rate","emp_length","addr_state","verification_status","purpose"]]

print("Model Dataset Shape:              ", model_data.shape)

print()

          #create new dataset with the suggested variables (dropped "policy code" beacause it is always 1)



model_data2 = pd.get_dummies(data=model_data,columns=["addr_state","term","verification_status","purpose", "emp_length"])

          #convert categorical data into dummies

    

feature_list2 = list(model_data2.columns)

          #saving list of features for later use



print("Dataset with Dummy Features Shape:", model_data2.shape)

          #size of dataset with dummies. 

          #Note that the number of observations remails at 1,303,638

          #The number of columns is now at 85 this compares to 9 in the "model_data" dataset

          #This makes sense given that we have 6 dummy features with 81 unique entries

          #9+81-5 = 85
#Split data

train_features2, test_features2, train_labels2, test_labels2 = train_test_split(model_data2, x2.target2, test_size=0.67, train_size=0.33, random_state=42)

print("Training Features Shape:",train_features2.shape)

print("Testing Features Shape: ",test_features2.shape)

print("Training Labels Shape:  ", train_labels2.shape)

print("Testing Labels Shape:   ", test_labels2.shape)

        #split data using a training size of 33% (test_size corresponts) and random state 42

        #notie the the lables (y) is taken from the original dataset "x2" and the features comes from "model_data2"

        #print shape of training and test set.

        #Notice that the feature test set has 873,438 rows which corresponds to 67% of 1,303,638.

        

print()

print("Repaid Loans in Test Set:", test_labels2.sum())

        #Note that there are 698,154 repaid loans in the test data.

        #Therefore, a model would need to perform better than 

        #79.93% accuracy (1 - 698,154/873,438) to beat a "dumb" approach
train_features2, test_features2, train_labels2, test_labels2 = train_test_split(model_data2, x2.target2, test_size=0.67, train_size=0.33, random_state=42)

print("Training Features Shape:",train_features2.shape)

print("Testing Features Shape: ",test_features2.shape)

print("Training Labels Shape:  ", train_labels2.shape)

print("Testing Labels Shape:   ", test_labels2.shape)

        #split data using a training size of 33% (test_size corresponts) and random state 42

        #notie the the lables (y) is taken from the original dataset "x2" and the features comes from "model_data2"

        #print shape of training and test set.

        #Notice that the feature test set has 873,438 rows which corresponds to 67% of 1,303,638.

        

print()

print("Repaid Loans in Test Set:", test_labels2.sum())

        #Note that there are 698,154 repaid loans in the test data.

        #Therefore, a model would need to perform better than 

        #79.93% accuracy (1 - 698,154/873,438) to beat a "dumb" approach
#Train model

rf = RandomForestClassifier(n_estimators = 100, max_depth = 4, random_state = 42, class_weight = "balanced")

       #classify ... To correct for the unbalanced data I weighted the sampling 

       #The “balanced” mode uses the values of y to automatically adjust weights 

       #inversely proportional to class frequencies in the input data (default/repaid)

    

rf.fit(train_features2,train_labels2)

       #fit the model
#Model outputs

predictions2 = rf.predict(test_features2)

      #make predictions in the test sample

      #note that the predictor takes the probability and rounds them to be either 0 or 1



print("Types of Prediction:", np.unique(predictions2))

print()

      #confirm that predictions are binary



print("Accuracy:",round(metrics.accuracy_score(test_labels2, predictions2)*100,4),"%")

print()

       #accuracy of 64.7681%

       #Therfore, the model performs worse that guessing that all will repay

        

#show proportion of loans repaid in test data

print("Repaid Loans in Test Set %:    ", round((test_labels2.sum() / test_labels2.size)*100,4),"%")

print()

       #Notice that the accuracy of the model is worse than the 79.93% of repaid 

       #loans in the test data. We would be better off projecting that all loans would be repaid



#Generate confusion matrix

def print_confusion_matrix(test_labels2,predictions2):

    cm = confusion_matrix(test_labels2,predictions2)

    print("Defaulting Applicant Rejected = ", cm[0][0])

    print("Defaulting Applicant Approved = ", cm[0][1])

    print("Paying Applicant Rejected =     ", cm[1][0])

    print("Paying Applicant Approved =     ", cm[1][1])



print("Confusion Matrix:")

print_confusion_matrix(test_labels2,predictions2)