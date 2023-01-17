# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sklearn #machine learning

import sklearn.model_selection as sklearn_model_selection #machine learning

import sklearn.ensemble as sklearn_ensemble #machine learning



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Create two new dataframes from Model Trap Loan Data



test = pd.read_csv("../input/test.csv", low_memory=False)

train = pd.read_csv("../input/train.csv", low_memory=False)
#Question 1

#What percentage of your training set loans are in default?



#Retrieve and print the percentage of instances where defalut = true

print('Question 1:','\n', '\n','Percentage of training set loans in default:', train.default.value_counts(True)[1]*100, '%')



#How does this work? How does it know that True is 1?
#Question 2



#Which ZIP code has the highest default rate in the training dataset?



default_by_zip=train.default.groupby(train.ZIP).mean()

#New object to store mean default rate grouped by zip



print('Question 2','\n', '\n','ZIP code with highest default rate:', default_by_zip.idxmax())

#idxmax function retrieves highest Zip Code
#Question 3



#What is the default rate in the training set for the first year for which you have data?



print('Question 3','\n','\n', 'Default rate in the first year for which we have data:', train.default.groupby(train.year).mean()[0]*100, '%')

#Group mean default by year and print 1st year 

#Order of years is in ascending order by default therefore specifying [0] will call 1st year
#Question 4



#What is the correlation between age and income in the training dataset? You may choose to use the correlation function .corr 



#Use corr function to determine correlation between income and age



print('Question 4','\n', '\n', 'Correlation between age and income:', train['age'].corr(train['income'])*100, '%')



#How do we know which comes first? - Slight difference
#Question 5



#What is the in-sample accuracy? 

#That is, find the accuracy score of the fitted model for predicting the outcomes using the whole training dataset.



y_train = train['default']

#Create training outcome dataset



x_train = train[['rent', 'education', 'income', 'loan_size', 'payment_timing' , 'job_stability', 'ZIP', 'occupation']]

#Create data set of features which will be used to predict default value and be comared against y_train above

#Order of predictive features will influence the accuracy of the model



x_train = pd.get_dummies(x_train)

#Convert all categorical variables in 'x_train' into dummy variables using function 'pd.get_dummies' - need to change this
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, oob_score=True, n_jobs=-1)

#Create object 'clf' based on the hyper parameters specified.



#Assumed maximum depth of 4 to avoid model overfitting. 

#This may cause reuslts to differ slightly from code used in homework 2 sample solutions.

clf.fit(x_train, y_train.values.ravel())

#Fit the model using training data, with ‘default’ as outcome variable, and save to 'clf' object
print('Question 5','\n', '\n', 'In-sample accuracy:', sklearn.metrics.accuracy_score(clf.predict(x_train), y_train)*100, '%')

#Accuracy of in-sample prediction calculated 

#This shows how well model predicts outcomes using data it was trained on
#Question 6

#What is the out of bag score for the model?

    

print('Question 6','\n', '\n', 'Out of bag score:', clf.oob_score_*100, '%')

#Calculate out-of-bag score for the fitted model 'clf'
#Question 7

#What is the out of sample accuracy?



y_test = test[['default']]

#New object 'y_test' that contains only default outcomes for test data 



x_test = test[['rent', 'education', 'income', 'loan_size', 'payment_timing' , 'job_stability', 'ZIP', 'occupation']]

#New object 'x_test' that contains only predictive features as used in Q5



x_test = pd.get_dummies(data = x_test)

#Convert all categorical variables in 'x_test' into dummy variables using function 'pd.get_dummies' - need to change this
out_sample_pred = clf.predict(x_test)

#New object to store out-of-sample prediction using the previously trained 'clf' model
print('Question 7','\n', '\n', 'Out-of-sample accuracy:', sklearn.metrics.accuracy_score(out_sample_pred, y_test)*100, '%')

#Accuracy of out-of-sample prediction calculated by comparing prediction based on test data with actual outcomes

#This shows how well model predicts outcomes using data it was not trained on
#Question 8

#What is the predicted average default probability for all non-minority members in the test set? 



test['out_sample_pred'] = (out_sample_pred)

#New column in 'test data' which stores predicted default



minority_default = test.out_sample_pred.groupby(test.minority).mean()

#New object to store model predictions for default outcome grouped by minority status



print('Question 8','\n', '\n', 'Default rate for non-minorities:', minority_default[0]*100, '%')

#Default outcomes for non-minorities.
#Question 9

#What is the predicted average default probability for all minority members in the test set?



print('Question 9','\n', '\n', 'Default rate for minorities:', minority_default[1]*100, '%')

#Default outcomes for minorities.
#Quesiton 10

#Is the loan granting scheme (the cutoff, not the model) group unaware? 



print('Question 10','\n', '\n', 'The cut-off is group unaware as it applies the same cut-off of 50% to all groups in the data set ')



#Group unaware reference material: https://research.google.com/bigpicture/attacking-discrimination-in-ml/ 

#Question 11

#Has the loan granting scheme achieved demographic parity? 



female = test[(test.sex == 1)]

male = test[(test.sex == 0)]

minority = test[(test.minority == 1)]

non_minority= test[(test.minority == 0)]

#Split out data into relevant demographicgroups (female, male, minority or non-minority)

#Demographic parity heuristic: Fraction of approved applications are the same across demographic groups 


print('Question 11','\n', '\n')

print('Percentage of accepted and rejected - Minority applicants', '\n', '\n', minority.out_sample_pred.value_counts(True)*100,'\n')

print('Percentage of accepted and rejected - Non-minority applicants', '\n', '\n', non_minority.out_sample_pred.value_counts(True)*100, '\n')

print('Percentage of accepted and rejected - Female applicants', '\n', '\n', female.out_sample_pred.value_counts(True)*100, '\n')

print('Percentage of accepted and rejected - Male applicants', '\n', '\n', male.out_sample_pred.value_counts(True)*100, '\n')

#Ouput the percentage of accepted and rejected applicants for each group

#Accepted applicants = "False"; rejected applicants = "True"

print('For demographic parity to be true the fraction of applicants that are predicted not to default and therefore are granted loans should be the same across the various demographic groups.')

print('As can be seen by the output above that is not the case with each group receiving different acceptance rates')

print('The "positive rate" is therefore not in parity and so demographic parity has not been achieved.')

#Conclusion on demographic parity
#Question 12

#Is the loan granting scheme equal opportunity? 



#Question 12 of homework 2 asks students to calculate the share of successful applicants that defaulted for each group (i.e., the false positive rate).

#However, a more effective way of evaluating whether the loan granting scheme is equal opportunity is the true positive rate. 

#This is because the criterion of equal opportunity states that, among the applicants who can pay back a loan, the same fraction in each group should actually be granted a loan. 

#We show data on both the true positive rate and the false positive rate below, concluding that the loan granting scheme is in fact equal opportunity



#For more information on discrimination and machine learning, c.f. https://research.google.com/bigpicture/attacking-discrimination-in-ml/



#First, we use a confusion matrix to compare the model's prediction ('out_sample_pred') to actual outcomes for each group (male, female, minority, non-minority) 

#This gives us the count of true negatives, false negatives, true positives and false positives for each group.





tn_male, fp_male, fn_male, tp_male = sklearn.metrics.confusion_matrix(male.default, male.out_sample_pred,  labels=[1,0]).ravel()

#Confusion matrix for male applicants





tn_female, fp_female, fn_female, tp_female = sklearn.metrics.confusion_matrix(female.default, female.out_sample_pred,  labels=[1,0]).ravel()

#Confusion matrix for female applicants





tn_minority, fp_minority, fn_minority, tp_minority = sklearn.metrics.confusion_matrix(minority.default, minority.out_sample_pred,  labels=[1,0]).ravel()

#Confusion matrix for minority applicants



tn_non_minority, fp_non_minority, fn_non_minority, tp_non_minority = sklearn.metrics.confusion_matrix(non_minority.default, non_minority.out_sample_pred,  labels=[1,0]).ravel()

#Confusion matrix for non-minority applicants and saving values
#Next, in order to evaluate whether the loan granting scheme is equal opportunity, we compare the percentage of paying applicants that get loans across groups. 



print('True positive rate by group:', '\n')

print('minority', (tp_minority/(tp_minority+fn_minority))*100,'%')

print('non_minority', (tp_non_minority/(tp_non_minority+fn_non_minority))*100,'%', '\n')

print('female:', (tp_female/(tp_female+fn_female))*100,'%')

print('male', (tp_male/(tp_male+fn_male))*100, '%')
#Finally, we calculate the false positive rates the way question 12 is asked in homework 2. Differences across groups do not affect whether equal opportunity has been achieved.



print('False positive rate by group:', '\n')

print('minority', (fp_minority/(fp_minority+tp_minority))*100,'%')

print('non_minority', (fp_non_minority/(fp_non_minority+tp_non_minority))*100,'%', '\n')

print('female:', (fp_female/(fp_female+tp_female))*100,'%')

print('male', (fp_male/(fp_male+tp_male))*100, '%')
#Additional analysis for group presentation



#We calculate default rates across groups in the training and test data to see if there is sampling bias



train_bias=train.default.groupby(train.minority).mean()

test_bias=test.default.groupby(test.minority).mean()



print('Default rate for minorities', '\n')

print('Training data:', train_bias[1]*100, '%')

print('Testing data:', test_bias[1]*100, '%', '\n', '\n')



print('Default rate for non-minorities', '\n')

print('Training data:', train_bias[0]*100, '%')

print('Testing data:', test_bias[0]*100, '%')