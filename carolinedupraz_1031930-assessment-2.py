import numpy as np # linear algebra

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt #plotting library

import sklearn as sklearn

import sklearn.model_selection as sklearn_model_selection

import sklearn.ensemble as sklearn_ensemble

import seaborn as sns #visualization



import os

print(os.listdir("../input"))
#Create new array by reading data



test = pd.read_csv("../input/test.csv", low_memory=False)

train = pd.read_csv("../input/train.csv", low_memory=False)
train.head()

#train.shape
test.head()

test.shape
#Question 1



#Create new object 'loans_in_default' that has fraction of loans in default and fraction of loans not in default ('default' column contains values 'True' and 'False' respectively) 

loans_in_default = train.default.value_counts(True)



#We are interested in the percentage of observations where the value of 'default' is 'True'. Because entries are ordered alphabetically, this will be the 2nd entry in the series. We retrieve and print this value.

print('Question 1','\n', '\n','Percentage of training set loans in default:', loans_in_default[1]*100, '%')
#Question 2



#Group default rate ('default') by zip code ('ZIP') and save output in new series 'default_by_zip'



default_by_zip=train.default.groupby(train.ZIP).mean()



print('Question 2','\n', '\n')

print('ZIP code with highest default rate:', default_by_zip.idxmax())
#Question 3



#Group default rate ('default') by year ('year') and save output in new series 'default_by_year'



default_by_year = train.default.groupby(train.year).mean()



#By default, 'default_by_year' is sorted in numerical order. Since we are interested in the first year, we retrieve the value of the first row of 'default_by_year'



print('Question 3','\n', '\n')

print('Default rate in the first year for which we have data:', default_by_year[0]*100, '%')
#Question 4



#Calculate the correlation between 'income' and 'age'



print('Question 4','\n', '\n')

print('Correlation between age and income:', train['income'].corr(train['age'])*100, '%')
#Question 5



#Create new object 'y_train' that contains only default status



y_train = train[['default']]



#Create new object 'x_trainy_te' that contains only predictive features (e.g. drop 'default' from data)



x_train = train[['ZIP','occupation','rent','education','income','loan_size','payment_timing','job_stability']]



#Convert all categorical variables in 'train_x' into dummy variables using function 'pd.get_dummies'



x_train = pd.get_dummies(data = x_train)
#Specify random forest classifier model to object 'clf' based on hyper parameters specified. Assume maximum depth of 4 to avoid overfitting.



clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, oob_score=True, n_jobs=-1)
#Fit random forest classifier model using training data, with ‘default’ as outcome variable, and save to 'clf' object



clf.fit(x_train, y_train.values.ravel())
# Create an in-sample prediction using previously trained 'clf' model



in_sample_pred = clf.predict(x_train)



#Calculate and display in-sample accuracy by comparing outcome of 'in_sample_pred' with original 'y_train' data



print('Question 5','\n', '\n')

print('In-sample accuracy:', sklearn.metrics.accuracy_score(in_sample_pred, y_train)*100, '%')
#Question 6

    

#Retrieve out-of-bag score for the fitted model 'clf'



print('Question 6','\n', '\n')

print('Out of bag score for the fitted model:', clf.oob_score_*100, '%')
#Question 7



#Create new object 'y_test' that contains only default outcomes for test data



y_test = test[['default']]



#Create new object 'x_test' that contains only predictive features (e.g. drop 'default' from data) for test data



x_test = test[['ZIP','occupation','rent','education','income','loan_size','payment_timing','job_stability']]



#Convert all categorical variables in 'x_test' into dummy variables using function 'pd.get_dummies'



x_test = pd.get_dummies(data = x_test)



# Create an out-of-sample prediction using previously trained 'clf' model



out_sample_pred = clf.predict(x_test)
#Calculate and display in-sample accuracy by comparing outcome of 'out_sample_pred' with original 'y_test' data



print('Question 7','\n', '\n')

print('Out-of-sample accuracy:', sklearn.metrics.accuracy_score(out_sample_pred, y_test)*100, '%')
#Question 8



#Create new column in test data ('out_sample_pred') that contains model predition for default outcome



test['out_sample_pred'] = (out_sample_pred)



#Group model predictions for default outcome ('test.out_sample_pred') by minority status and save to new object 'minority_default'



minority_default = test.out_sample_pred.groupby(test.minority).mean()



#Retrieve and display default outcomes for non-minorities



print('Question 8','\n', '\n')

print('Default rate for non-minorities:', minority_default[0]*100, '%')
#Question 9



print('Question 9','\n', '\n')

print('Default rate for minorities:', minority_default[1]*100, '%')
#Quesiton 10



print('Question 10','\n', '\n')

print('The loan granting scheme is group unaware. The model uses the same cut-off (50%) for all groups after calculating the default probability of each applicants')
#Question 11



#In order to evaluate whether demographic parity has been achieved, we compare the share of approved applicants to the share of rejected applicants for women and minority groups. 

#We assume that an approved female applicant is one which the model predicts will not default, and a rejected female applicant is one that the model predicts will default. 

#We compare these to the ratio of approved-to-rejected applicants for non-female and non-minority applicants.



#First, we create four new dataframes that contain only the observations relevant for a particular group (e.g. each data set contains either all minority, non-minority, female, or male applicants)



female = test[(test.sex == 1)]

minority = test[(test.minority == 1)]

male = test[(test.sex == 0)]

non_minority= test[(test.minority == 0)]



#Next, we calculate the percentage of accepted and rejected applicants for each dataframe and compare them.



print('Question 11','\n', '\n')

print('Percentage of accepted and rejected - Minority applicants', '\n', '\n', minority.out_sample_pred.value_counts(True)*100,'\n')

print('Percentage of accepted and rejected - Non-minority applicants', '\n', '\n', non_minority.out_sample_pred.value_counts(True)*100, '\n')

print('Percentage of accepted and rejected - Female applicants', '\n', '\n', female.out_sample_pred.value_counts(True)*100, '\n')

print('Percentage of accepted and rejected - Male applicants', '\n', '\n', male.out_sample_pred.value_counts(True)*100, '\n')



#We print our conclusions

print('The criterion of demographic parity allows us to see whether the fraction of applicants getting loans is the same across groups. The model estimates substantially higher default rates for minority applicants compared to non-minority applicants. Therefore, if we use the predicted default status as an indicator for whether a loan was made or not, then the "positive rate" is different across these groups. No difference is observed between male and female applicants.')
#Question 12



#In order to evaluate whether equal opportunity has been achieved, we compare the share of non-paying that get loans across various groups.



#First, we create two new dataframes that contain only the observations related to male applicants and non-minority applicants, respectively



male = test[(test.sex == 0)]

non_minority= test[(test.minority == 0)]
#For each group (male, female, minority, non-minority), we use a confusion matrix to evaluate the accuracy of the model's classification of whether or not someone in that group would default ('out_sample_pred'). 

#This gives us the count of true negatives, false negatives, true positives and false positives for each group.



#Creating confusion matrix for male applicants and saving values

tn_male, fp_male, fn_male, tp_male = sklearn.metrics.confusion_matrix(male.default, male.out_sample_pred).ravel()



#Creating confusio matrix for female applicants and saving values

tn_female, fp_female, fn_female, tp_female = sklearn.metrics.confusion_matrix(female.default, female.out_sample_pred).ravel()



#Creating confusion matrix for minority applicants and saving values

tn_minority, fp_minority, fn_minority, tp_minority = sklearn.metrics.confusion_matrix(minority.default, minority.out_sample_pred).ravel()



#Creating confusion matrix for non-minority applicants and saving values

tn_non_minority, fp_non_minority, fn_non_minority, tp_non_minority = sklearn.metrics.confusion_matrix(non_minority.default, non_minority.out_sample_pred).ravel()
# We compare the shares of successful applicants that defaulted between minority and non-minority groups. We repeat the comparison for female and male applicants.



print('Question 12','\n', '\n')

print('minority', fn_minority/len(minority)*100,'%')

print('non_minority', fn_non_minority/len(non_minority)*100,'%', '\n')

print('female:', fn_female/len(female)*100,'%')

print('male', fn_male/len(male)*100, '%','\n', '\n')



print('No, the loan granting scheme is not equal opportunity. The percentage of successful non-minority applicants that defaulted is significantly higher than the share of minority applicants that default. Among groups that secured a loan, the likelihood of default is higher among the non-minority group than among minority group. In other words, it is easier to secure a loan if you are a non-minority applicant.')