# We begin by loading several helpful packages into this Python 3 environment



import numpy as np # linear algebra

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sklearn #machine learning

import sklearn.model_selection as sklearn_model_selection #machine learning

import sklearn.ensemble as sklearn_ensemble #machine learning



# Input data files are available in the "../input/" directory. Running this code lists the files in the input directory



import os

print(os.listdir("../input"))
#Create two new dataframes by reading data



test = pd.read_csv("../input/test.csv", low_memory=False)

train = pd.read_csv("../input/train.csv", low_memory=False)
#Question 1



#We calculate fraction of loans in default and fraction not in default, and save output to new obejct 'loans_in_default' 

loans_in_default = train.default.value_counts(True)



#We are interested in the share of observations where the value of 'default' is 'True'. This will be the 2nd row in 'loans_in_default'. We retrieve and print this value.

print('Question 1:','\n', '\n','Percentage of training set loans in default:', loans_in_default[1]*100, '%')
#Question 2



#We group default rate ('default') by zip code ('ZIP') and save output in new object 'default_by_zip'



default_by_zip=train.default.groupby(train.ZIP).mean()



#We retrieve the ZIP code with the highest default rate using idxmax



print('Question 2','\n', '\n')

print('ZIP code with highest default rate:', default_by_zip.idxmax())
#Question 3



#We group default rate ('default') by year ('year') and save output in new object 'default_by_year'



default_by_year = train.default.groupby(train.year).mean()



#By default, 'default_by_year' will be sorted in numerical order. Since we are interested in the earliest year, we retrieve the value of the first row of 'default_by_year'



print('Question 3','\n', '\n')

print('Default rate in the first year for which we have data:', default_by_year[0]*100, '%')
#Question 4



#We calculate the correlation between 'income' and 'age'



print('Question 4','\n', '\n')

print('Correlation between age and income:', train['income'].corr(train['age'])*100, '%')
#Question 5



#We wish to train a Random Forest Classifier model using our training data. First, we prepare our training data.



#Create new object 'y_train' that contains only default status



y_train = train['default']



#Create new object 'x_train' that contains only predictive features specified in homework assignment (e.g. drop 'default' from data, along with protected characteristics such as "minority", "sex", "age", and "year")

#Note that the order specified for the predictive features will influence the accuracy of the model (though to a limited extent).



x_train = train[['rent', 'education', 'income', 'loan_size', 'payment_timing' , 'job_stability', 'ZIP', 'occupation']]



#Convert all categorical variables in 'train_x' into dummy variables using function 'pd.get_dummies'



x_train = pd.get_dummies(x_train)
#Next, we specify a random forest classifier model to object 'clf' based on the hyper parameters specified.



#I assume a maximum depth of 4 to avoid model overfitting. Note that this may cause my reuslts to differ slightly from code used in homework 2 sample solutions.



clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, oob_score=True, n_jobs=-1)
#We fit the model using training data, with ‘default’ as outcome variable, and save to 'clf' object



clf.fit(x_train, y_train.values.ravel())
#We create an in-sample prediction in order to see how well our model predicts the data it was trained on (clf.predict(x_train)). 

#We then calculate the accuracy of that in-sample prediction and display the result.



print('Question 5','\n', '\n')

print('In-sample accuracy:', sklearn.metrics.accuracy_score(clf.predict(x_train), y_train)*100, '%')
#Question 6

    

#We retrieve out-of-bag score for the fitted model 'clf'



print('Question 6','\n', '\n')

print('Out of bag score:', clf.oob_score_*100, '%')
#Question 7 



#We wish to use our fitted model to create an out-of-sample forecast. First, we prepare our testing data.



#We create a new object 'y_test' that contains only default outcomes for test data 



y_test = test[['default']] 



#We create a new object 'x_test' that contains only predictive features (e.g. drop 'default' from data as well as previously discussed protected characteristics)



x_test = test[['rent', 'education', 'income', 'loan_size', 'payment_timing' , 'job_stability', 'ZIP', 'occupation']]



#We convert all categorical variables in 'x_test' into dummy variables using function 'pd.get_dummies' x_test = pd.get_dummies(data = x_test)



x_test = pd.get_dummies(data = x_test)
#Next, we create an out-of-sample prediction using the previously trained 'clf' model. We save the results of this prediction to a new object 'out_sampe_pred'

out_sample_pred = clf.predict(x_test)
#We calculate the accuracy of the out-of-sample prediction by comparing the model's predictions ('out_sample_pred') with the original default data ('y_test')



print('Question 7','\n', '\n')

print('Out-of-sample accuracy:', sklearn.metrics.accuracy_score(out_sample_pred, y_test)*100, '%')
#Question 8



#We create a new column in the original test data (also named 'out_sample_pred') that contains the model's prediction for default



test['out_sample_pred'] = (out_sample_pred)



#We group model predictions for default outcome ('test.out_sample_pred') by minority status and save to new object 'minority_default'



minority_default = test.out_sample_pred.groupby(test.minority).mean()



#We retrieve and display the default outcomes for non-minorities (first line)



print('Question 8','\n', '\n')

print('Default rate for non-minorities:', minority_default[0]*100, '%')
#We retrieve and display the default outcomes for minorities (second line)



print('Question 9','\n', '\n')

print('Default rate for minorities:', minority_default[1]*100, '%')
#Quesiton 10



#For more information on discrimination and machine learning, c.f. https://research.google.com/bigpicture/attacking-discrimination-in-ml/ 



print('Question 10','\n', '\n')

print('The loan granting scheme is group unaware. The model calculates the default probability of each applicants and then applies the same cut-off (50%) to all groups')
#Question 11



#For more information on discrimination and machine learning, c.f. https://research.google.com/bigpicture/attacking-discrimination-in-ml/ 



#In order to evaluate whether demographic parity has been achieved, we compare the share of approved applicants and rejected applicants across gender and minority groups. 



#First, we create four new dataframes that contain only the observations relevant to each group (e.g. each data set contains either all minority, non-minority, female, or male applicants)



female = test[(test.sex == 1)]

male = test[(test.sex == 0)]

minority = test[(test.minority == 1)]

non_minority= test[(test.minority == 0)]
#Next, we calculate and display the percentage of accepted and rejected applicants for each group and compare them. 

#Accepted applicants are those for whom the predicted value is "False" and rejected applicants are those with a predicted value "True"



print('Question 11','\n', '\n')

print('Percentage of accepted and rejected - Minority applicants', '\n', '\n', minority.out_sample_pred.value_counts(True)*100,'\n')

print('Percentage of accepted and rejected - Non-minority applicants', '\n', '\n', non_minority.out_sample_pred.value_counts(True)*100, '\n')

print('Percentage of accepted and rejected - Female applicants', '\n', '\n', female.out_sample_pred.value_counts(True)*100, '\n')

print('Percentage of accepted and rejected - Male applicants', '\n', '\n', male.out_sample_pred.value_counts(True)*100, '\n')
#Based on the data above, we come to the following conclusion.



print('The criterion of demographic parity allows us to examine whether the fraction of applicants getting loans is the same across groups.')

print('As the above data shows, the model estimates substantially higher default rates for minority applicants (4.6%) compared to non-minority applicants (0.1%).')

print('We also observe a discrepancy between female (2.8%) and male applicants (1.9%), though to a lesser degree.')

print('Differences in the “positive rate” across groups indicates that the loan granting scheme is not making loans to each group at the same rate.')

print('This means that the criteria of demographic parity has not been achieved.')
#Question 12



#Question 12 of homework 2 asks students to calculate the share of successful applicants that defaulted for each group (i.e., the false positive rate).

#However, a more effective way of evaluating whether the loan granting scheme is equal opportunity is the true positive rate. 

#This is because the criterion of equal opportunity states that, among the applicants who can pay back a loan, the same fraction in each group should actually be granted a loan. 

#We show data on both the true positive rate and the false positive rate below, concluding that the loan granting scheme is in fact equal opportunity



#For more information on discrimination and machine learning, c.f. https://research.google.com/bigpicture/attacking-discrimination-in-ml/



#First, we use a confusion matrix to compare the model's prediction ('out_sample_pred') to actual outcomes for each group (male, female, minority, non-minority) 

#This gives us the count of true negatives, false negatives, true positives and false positives for each group.



#Creating confusion matrix for male applicants and saving values

tn_male, fp_male, fn_male, tp_male = sklearn.metrics.confusion_matrix(male.default, male.out_sample_pred,  labels=[1,0]).ravel()



#Creating confusion matrix for female applicants and saving values

tn_female, fp_female, fn_female, tp_female = sklearn.metrics.confusion_matrix(female.default, female.out_sample_pred,  labels=[1,0]).ravel()



#Creating confusion matrix for minority applicants and saving values

tn_minority, fp_minority, fn_minority, tp_minority = sklearn.metrics.confusion_matrix(minority.default, minority.out_sample_pred,  labels=[1,0]).ravel()



#Creating confusion matrix for non-minority applicants and saving values

tn_non_minority, fp_non_minority, fn_non_minority, tp_non_minority = sklearn.metrics.confusion_matrix(non_minority.default, non_minority.out_sample_pred,  labels=[1,0]).ravel()
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