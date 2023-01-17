import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
#Load data from CSV file

data = pd.read_csv("../input/loan.csv", low_memory = False)
#Truncate the data to remain only with those fully paid and those that defaulted

data = data[(data.loan_status == 'Fully Paid') | (data.loan_status == 'Default')]

# Add a target column with 1 for fully paid and 0 for defaulted loans

data['target'] = (data.loan_status == 'Fully Paid')
#question 1: Number of records is 1041983. There are 146 features (columns).

print('Number of records: '+str(data.shape[0])+'\nNumber of features: '+str(data.shape[1])) 
#Question 2



#Import library to be used for ploting

from matplotlib import pyplot as plt



#Question 2 a. plot histogram for loan amounts.

plt.hist(data.loan_amnt, bins=20)
#Question 2 b.

print('Statistical values of loan amounts column')

print("Mean: "+str(data.loan_amnt.mean())) #Mean

print("Median: "+str(data.loan_amnt.median())) #Median

print("Maximum: "+str(data.loan_amnt.max())) #Maximum

print("Standard deviation: "+str(data.loan_amnt.std())) #Standard deviation
#Question 3. Short and long term loans



#Function outputs means and standard deviations depending on term given via parameter list

def print_stats(term):

    #Get short term loan and long term loan dataframes

    if term == ' 36 months': ds= data[data.term==' 36 months']

    else: ds = data[data.term==' 60 months']



    #print rates mean and standard deviation

    print(term + " term rate Mean: %10.4f" %(ds.int_rate.mean()))

    print(term + " term rate Standard Deviation: %10.4f" %(ds.int_rate.std()))





#Make calls to the print_stats function to help print means and stds

print_stats(' 36 months')

print_stats(' 60 months')
#Question 3b

#Using seaborn library to plot because it is well colour coded

import seaborn as sns

sns.boxplot(x="term", y="int_rate", data=data)
#Question 4

#Visualise the data in question

data.boxplot('int_rate',by=('grade'))



#Print average rates for all grades to see which one has highest everage rate

print(data.int_rate.groupby(data.grade).mean())



#Collect grades as keys into a dictionary with their rate means as values

grade_rates = data.int_rate.groupby(data.grade).mean()



#Simply print the key with highest value together with its value

print('The grade with highest average rate is '+grade_rates.idxmax()+' with rate '+str(grade_rates.max()))
#Question 5

#Exploring relationship between grades interest rates and default rates 

data.grade.unique()

for grade in data.grade.unique():

    print('Grade: '+grade+' Interest rate: %7.4f' %(data[data.grade==grade].int_rate.mean())+' Default rate: %8.7f' %(((1-data[data.grade==grade].target.mean())*100))+'%')

    

print('It is interesting to observe that the grade with highest interest rate has the lowest default rate')
#Question 5 a

#Calculate realised yield for each grade

grade_realized_yield = data.total_pymnt.groupby(data.grade).sum()/data.funded_amnt.groupby(data.grade).sum()-1



#Print all grades with their realised yields

print(grade_realized_yield)



#Print key with highest value together with its value

print('The grade with highest realized yield is '+grade_realized_yield.idxmax()+' with realizeed yield '+str(grade_realized_yield.max()))
#Question 6 a

#See how many applications types are there

print('Application types are: '+str(data.application_type.unique()))



#Print the number of applications per application type

print(data['application_type'].value_counts())



#Question 6 b

data.groupby(['application_type', 'target']).count()['loan_amnt']



#Using the feature does not make much sense because the number of defaults per application type does not necessarily increase with application type

#Second, individual applications are much more than joint applications.
data.head()
#Question 7

#List of features we need to convert to dummies

dummy_list = ['term','emp_length','addr_state','verification_status','purpose']

#Create new dataset with features that we regard important for the model

important_features =data[['loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate','emp_length','addr_state','verification_status','purpose','policy_code']]

#Convert categorical features into dummies

data_set = pd.get_dummies(data = important_features, columns= dummy_list,sparse=True)



print('Total width of feature set is: '+str(data_set.shape[1]))
data_set.head()
#Question 8

from sklearn.model_selection import train_test_split

#Question 8

#Spliy the dataset set into training and testing sets

Y = np.array(data['target'])

X_train, X_test, y_train, y_test = train_test_split(data_set,Y, train_size=0.33, test_size=0.67, random_state=42)

print('X_train shape is as follows: rows: ', (X_train.shape[0]), ' columns: ', (X_train.shape[1]))
#Question 9

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



#No need to split the data here again because we are using data from question 8 

#which is already split according to quiestion 9 instructions



# Specify parapeters for random forest model

rf_model = RandomForestClassifier(n_estimators = 100, max_depth=4, random_state = 42, n_jobs=-1)



# Train the model using training data

rf_model.fit(X_train, y_train)



#Test

predictions = rf_model.predict(X_test)



print('Out of sample accuracy is: %7.4f' %(accuracy_score(y_test, predictions)*100)+'%')
#Visualise predictions through confusions matrix



#We are going to need confusion matrix several times, setting up a function for that is ideal

def confusion_matrix(ground_truth, predictions):

    from sklearn.metrics import confusion_matrix

    confusion_m=pd.DataFrame(confusion_matrix(ground_truth, predictions, labels=[1,0]), index=['true:1', 'true:0'], columns=['pred:1', 'pred:0'])

    print(confusion_m)



confusion_matrix(y_test, predictions)

print('\nFlase positive rate is 100% while false negative rate is 0% \n'  

      'This indicates how biased the model is in favor of positives \n')

#See the percentage of loan repayments compared to percentage of defaults

percentage_train = (y_train.sum()/len(y_train))*100



print('Percentage of repayments in training set is: %.5f' %(percentage_train)+'%')

print('Percentage of defaults in training set is: %.5f ' %(100- percentage_train)+'%')



print('The difference in amount of repayments training examples explains the imbalanced predictions')
#Question 10

#Build an array with all elements corresponding to True (ones) to indicate that everyone will pay

all_repmnt_pred= np.array([1]*len(y_test))



#Compare the new list with predictions generated by model

print("Accuracy on all data repayment is:  %7.4f" %(accuracy_score(y_test, all_repmnt_pred)*100)+"%")

print('Out of sample accuracy is: %7.4f' %(accuracy_score(y_test, predictions)*100)+'%')



#Accuracy this time is 99.9968%, exactly like in the first instance

#Which means all the models is predicting are repayments. This is is because the dataset is not balanced. 

#It has many repayments so the model built indicates thateveryone will not default and is not able 

#to learn characteristics of those that will default