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
#Crearting the dataset

data = pd.read_csv("../input/loan.csv", low_memory = False)

#Removing the loans that are still being repayed

data = data[(data.loan_status == 'Fully Paid') | (data.loan_status == 'Default')]

#Creating the variable of interest - whether the loan has been repayed

data['target'] = (data.loan_status == 'Fully Paid')
#Q1

data.shape

#There are 1041983 records and 146 features



data['target'].head()

len(data['target'])

import matplotlib as plt



#Studying our data

plt.pyplot.hist(data.loan_amnt, bins = 100)







a= data.loan_amnt.mean()

b= data.loan_amnt.median()

c =data.loan_amnt.max()

d= data.loan_amnt.std()



print ('Mean is', a, 'Median is', b, ' Maximum is', c,'Std is', d)



#Question 3

#a - splitting the sample depending on short/long duration

data_36 = data[(data.term == ' 36 months')] 

data_60 = data[(data.term == ' 60 months')] 







print(

data_36.int_rate.mean(),

data_36.int_rate.std(),

data_60.int_rate.mean(),

data_60.int_rate.std(),

)



#b

data.boxplot(column='int_rate', by = 'term')

#Q4 - looking at the interest rate for the lowest credit rating

data.boxplot(column='int_rate', by = 'grade')

print(data.int_rate[(data.grade == 'G')].mean())
#Question 5 - realized yield by grade:

(data.groupby(by='grade')['total_pymnt'].sum()/data.groupby(by='grade')['funded_amnt'].sum())-1 
#Grade F is the highest with the avg realized yield of  

print(max((data.groupby(by='grade')['total_pymnt'].sum()/data.groupby(by='grade')['funded_amnt'].sum())-1 ))
#Question 6 - Individual / Joint apps

len(data[data.application_type == 'Joint App' ]) # 17515

len(data[data.application_type == 'Individual' ]) #1024468



#Does not really make sense to use this characteristic because the split is 99.9:0.01, will have low accuracy

#Question 7

#Converting categorical variables to dummies

dterm = pd.get_dummies (data['term'],drop_first=True)

dadd = pd.get_dummies (data['addr_state'],drop_first=True)

demp = pd.get_dummies (data['emp_length'],drop_first=True)

dver = pd.get_dummies (data['verification_status'],drop_first=True)

dpur = pd.get_dummies (data['purpose'],drop_first=True)

    

#Creating the "X" matrix - our explanatory variables

data_dum= pd.concat([dterm,dadd,demp,dver,dpur, data.loan_amnt, data.funded_amnt, data.funded_amnt_inv,data.int_rate,data.policy_code],axis=1) #concatinating data

data_dum.shape

#[1041983 rows x 81 columns] - we excluded "base" values for each dummy variable

#Question 8 - randomly splitting the sample into 33% train, 67% test

# creating a variable for the "target" column

Y = data.target



#applying the splitter fumnction using the suggested specifications

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test= train_test_split(data_dum, Y, train_size = 0.33 , random_state=42) 

#Shape is [343854 rows x 81 columns]

#Question 9



#Estimating the RFC 

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)

clf.fit(X_train,Y_train) 



# Using the model to predict "target" observations for the "test" subsample

predicted_target = clf.predict(X_test)
#Using accuracy_score function to assess the accuracy of the prediction



from sklearn import metrics 

metrics.accuracy_score(Y_test, predicted_target) #0.9999684871993572
#Question 10



#Comparing our RandForClass predictor output to a vector of 1's (ie "everyone repays")

Y10 = np.ones(len(predicted_target))

metrics.accuracy_score(Y_test, Y10)



#Identical - suggests that a simple "everyone will repay" prediction is as good as the model we created
#Question 11 - Bonus 

datab =data[['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term','int_rate', 

                  'emp_length', 'addr_state','verification_status', 'purpose', 'policy_code', 'target' ]]

target_count = datab.target.value_counts()

print('Class 1:', target_count[0])

print('Class 0:', target_count[1])

# Only 31 defaulted observations, therefore we will use oversampling. undersampling will result in 62 total observations
#Over-sampling our dataset



# Divide by class

df_class_1 = datab[datab['target'] == 0]

df_class_0 = datab[datab['target'] == 1]

#Counting obs in each

#class 1 = Defaulted loan

count_class_0, count_class_1 = datab.target.value_counts()



#Increasing the number of defaulted observations

df_class_1_over = df_class_1.sample(count_class_0, replace=True)

df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)
#Checking that the over-sampling ran correctly:

print('Random over-sampling:')

print(df_test_over.target.value_counts())



df_test_over.target.value_counts().plot(kind='bar', title='Count (target)');
# Re-running our RFC model on the new sample

Yb = df_test_over.target

dtermb = pd.get_dummies (df_test_over['term'],drop_first=True)

daddb = pd.get_dummies (df_test_over['addr_state'],drop_first=True)

dempb = pd.get_dummies (df_test_over['emp_length'],drop_first=True)

dverb = pd.get_dummies (df_test_over['verification_status'],drop_first=True)

dpurb = pd.get_dummies (df_test_over['purpose'],drop_first=True)

    

#Creating the "X" matrix - our explanatory variables

Xb= pd.concat([dtermb,daddb,dempb,dverb,dpurb, df_test_over.loan_amnt, df_test_over.funded_amnt, df_test_over.funded_amnt_inv,df_test_over.int_rate,df_test_over.policy_code],axis=1) #concatinating data

Xb.head()

#Cutting the new sample

X_trainb, X_testb, Y_trainb, Y_testb= train_test_split(Xb, Yb, train_size = 0.33 , random_state=42) 

#Estimating coefficients

clfb = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)

clfb.fit(X_trainb,Y_trainb) 

# Using the model to predict "target" observations for the "test" subsample

ptb = clfb.predict(X_testb)
#Assessing the new model's accuracy

metrics.accuracy_score(Y_testb, ptb)

#84% now - significantly lower and more realistic than 99.9%