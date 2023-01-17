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
data = pd.read_csv("../input/loan.csv", low_memory = False)

data = data[(data.loan_status == 'Fully Paid') | (data.loan_status == 'Default')]

data['target'] = (data.loan_status == 'Fully Paid')
data.shape

data['target'].head()

len(data['target'])

import matplotlib as plt

plt.pyplot.hist(data.loan_amnt, bins = 100)



#data.hist(column='loan_amnt) 





a= data.loan_amnt.mean()

b= data.loan_amnt.median()

c =data.loan_amnt.max()

d= data.loan_amnt.std()



print (a,b,c,d)



data_36 = data[(data.term == ' 36 months')] 

data_60 = data[(data.term == ' 60 months')] 



data_36.int_rate.mean

data_60.int_rate.mean



data.boxplot(column='int_rate', by = 'term')

data.grade.unique()

data.boxplot(column='int_rate', by = 'grade')

data.int_rate[(data.grade == 'G')].mean()

#that's smart

data.int_rate.groupby(data.grade).mean() 



data.total_pymnt[(data.grade == 'G')]/data.loan_amnt[(data.grade == 'G')]

len(data[data.application_type == 'Joint App' ])

len(data[data.application_type == 'Individual' ])



data.application_type.unique()

#Question 7



#Categorical:

#term  emp_length addr_state verification_status purpose



#Convert to dummy variables 

dterm = pd.get_dummies (data['term'])

dadd = pd.get_dummies (data['addr_state'])

demp = pd.get_dummies (data['emp_length'])

dver = pd.get_dummies (data['verification_status'])

dpur = pd.get_dummies (data['purpose'])

    

#dum_dat = pd.concat([data.loan_amnt,  data.funded_amnt, data.funded_amnt_inv, dterm, data.int_rate, demp,dadd,dver,dpur,data.policy_code], axis = 1)1

#print (dum_dat) 

#data['dterm'] = pd.get_dummies (data['term'])

data_dum= pd.concat([dterm,dadd,demp,dver,dpur, data.loan_amnt, data.funded_amnt, data.funded_amnt_inv,data.int_rate,data.policy_code],axis=1) #concatinating data

print (data_dum) 

#[1041983 rows x 86 columns]



#Question 8 



from sklearn.model_selection import train_test_split

X_train, X_test= train_test_split(data_dum,train_size = 0.33 , test_size=0.67, random_state=42) 

print(X_train)

#Shape is [343854 rows x 86 columns]

#Question 9

#data_dum = pd.concat([target,data_dum])

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)

clf.fit(data_dum,data.target)

#target as outcome variable 

# n_estimators=100, max_depth=4  test_size=0.33 for splitting. Remember to set the random_state=42 for both splitting and training.