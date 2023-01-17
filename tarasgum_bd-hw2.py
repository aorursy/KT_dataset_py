# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import matplotlib as plt

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
#Creating the datasets

datatrain = pd.read_csv("../input/train.csv", low_memory = False)

datatest = pd.read_csv("../input/test.csv", low_memory = False)

#Question 1 - What % are in default



default_count = datatrain.default.sum()

default_percentage = default_count/len(datatrain)*100

print(default_percentage)

#print('Default percentage equals' pd.value_counts(data_train['default'].values, sort=False))
#Question 2 

#All 4 zipcodes % defaults

datatrain.groupby(by="ZIP").mean().default

#The Highest Default postcode

datatrain.groupby(by="ZIP").default.mean().idxmax()
#Question 3 - Default percentage in year 1

datatrain.default[datatrain.year== 0].mean()

#datatrain.head()
#Question 4 - correlation betwen age and income

datatrain['age'].corr(datatrain['income'])
datatest.head()
#Question 5 - What's the model accuracy in-sample

clf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score = True)



Y_train = datatrain.default

X_train = pd.concat([datatrain.drop( ['Unnamed: 0', 'default', 'occupation', 'ZIP', 'sex', 'minority'], axis=1), 

                     pd.get_dummies (datatrain['occupation']),pd.get_dummies (datatrain['ZIP'])], axis=1)
#Estimating the coefficients

clf.fit(X_train,Y_train)
#Looking at accuracy

clf.score(X_train, Y_train)

#Equal to 1! Not a single error in prediction
#Question 6 - Looking at the OOB score

clf.oob_score_

#0.9999666666666667 - slightly lower but still very high
#The score is 1 - suspicious. Checking with confusion matrix:

Y_pred = clf.predict(X_train)

from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_true=Y_train, y_pred=Y_pred)

print('Confusion matrix:\n', conf_mat)

#Seems like the model did ineed predict EVERYTHING correctly in-sample
#Question 7 - Checking the out-of-sample accuracy

Y_test = datatest.default

X_test = pd.concat([datatest.drop( ['Unnamed: 0','Unnamed: 0.1' ,'default', 'occupation', 'ZIP', 'sex', 'minority'], axis=1), 

                     pd.get_dummies (datatest['occupation']),pd.get_dummies (datatest['ZIP'])], axis=1)
clf.score(X_test, Y_test)

#73.9% - much lower than both OOB and in-sample acuracy
#Questions 8,9  - Predicted default probabilities split by minority/non minority



#Generating the array of predicted default stati

Y_oos_pred = clf.predict(X_test)

out = pd.Series(data=Y_oos_pred)

out=out.rename('defaults')
#Collecting the predictions & regressors & sex w/ minority in 1 dataset

dataq10= pd.concat([out, X_test, datatest.sex, datatest.minority],axis=1)
#Splitting dataq7 into a minority and non-minority samples. CHANGE - N-Minority - 15.4%, M - 35.8%

dataq10.groupby(by="minority").mean().defaults
#Question 10

#Yes it is group unaware, as the cutoff point (50%) is the same for every category/group.
#Question 11

#Checking demographic parities. Male - 18.5% rejected applicants, Female - 18.9%. 

#The gender demographic parity held - the same proportions of M & F applicants received loans.

#Ethnical didnjt - minorities got 6% approval rate while non-m got 31%;
#Looking for any interaction between minority status and gender:



dataq10.groupby(by=['sex', 'minority']).mean().defaults

#We can see that the model does not predict any additional increase in dfrault rates as a 

#result of the interaction between "minority" and "sex" - eg no further discrimination
#Question 12 - Equality of opportunity. We need to check that the same proportions of "loanworthy" 

#candidates was approved across the demographics



dataq12=pd.concat([out, datatest.default, datatest.sex, datatest.minority],axis=1)



#We then create a sub-sample "paid" - those applicants that repayed their loan. 

#We will later check whether the model correctly predicted that they will pay.

paid = dataq12[(dataq12.default==0)]
#Looking at predicted default rates among those applicants who actually did not default

paid.groupby(by='minority').mean().defaults



#We can see that the model would have incorrectly rejected 12% of "loanworthy" non-m and 30.5% of "loanworthy" minority applicants

#This clearly violates equality of opportunity.
#Looking at the same, but split by gender

paid.groupby(by='sex').mean().defaults

#Same effect present but of smaller magnitude - 20% of "loanworthy" males were rejected vs 23% of loanworthy females
#WHAT THE QUESTION LITERALLY ASKS TO DO (but is the opposite of what google.research says we should do)

approved = dataq12[(dataq12.defaults==0)]

approved.groupby(by='minority').mean().default

#approved.groupby(by='sex').mean().defaults