### load dataset & dependencies
import os,math,io
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
bureau_data = pd.read_csv('../input/bureau.csv')
train = pd.read_csv('../input/application_train.csv')
train_col = train.columns
emp_phone = train.FLAG_MOBIL==1
empp = train[emp_phone]
print(train.TARGET.value_counts(normalize=True))
train['DOCS'] = train['FLAG_DOCUMENT_2']+train['FLAG_DOCUMENT_3']+train['FLAG_DOCUMENT_4']+train['FLAG_DOCUMENT_5']+train['FLAG_DOCUMENT_6']+train['FLAG_DOCUMENT_7']+train['FLAG_DOCUMENT_8']+train['FLAG_DOCUMENT_9']+train['FLAG_DOCUMENT_10']+train['FLAG_DOCUMENT_11']+train['FLAG_DOCUMENT_12']+train['FLAG_DOCUMENT_13']+train['FLAG_DOCUMENT_14']+train['FLAG_DOCUMENT_15']+train['FLAG_DOCUMENT_16']+train['FLAG_DOCUMENT_17']+train['FLAG_DOCUMENT_18']+train['FLAG_DOCUMENT_19']+train['FLAG_DOCUMENT_20']+train['FLAG_DOCUMENT_21']
train.DOCS.unique()
for i in train.DOCS.unique():
    D1 = train.DOCS==i
    D3 = train[D1]
    print(i,D3.TARGET.value_counts(normalize=True),train.DOCS.mean())
train.NAME_CONTRACT_TYPE.fillna(0,inplace=True)
train.NAME_CONTRACT_TYPE.unique()
train.replace({'Cash loans', 'Revolving loans'},{1,2},inplace=True)
for i in train.NAME_CONTRACT_TYPE.unique():
    D1 = train.NAME_CONTRACT_TYPE==i
    D3 = train[D1]
    print(i,D3.TARGET.value_counts(normalize=True),D3.TARGET.value_counts())
train.NAME_INCOME_TYPE.unique()
train.replace({'Working', 'State servant', 'Commercial associate', 'Pensioner',
       'Unemployed', 'Student', 'Businessman', 'Maternity leave'},{1,2,3,4,5,6,7,8},inplace=True)
for i in train.NAME_INCOME_TYPE.unique():
    V1 = train.NAME_INCOME_TYPE==i
    V2 = train[V1]
    print(i,V2.TARGET.value_counts(),i,V2.TARGET.value_counts(normalize=True))
train.NAME_EDUCATION_TYPE.unique()
train.replace({'Secondary / secondary special', 'Higher education',
       'Incomplete higher', 'Lower secondary', 'Academic degree'},{1,2,3,4,5},inplace=True)
for i in train.NAME_EDUCATION_TYPE.unique():
    edu1 = train.NAME_EDUCATION_TYPE==i
    edu = train[edu1]
    print(i,edu.TARGET.value_counts(),i,edu.TARGET.value_counts(normalize=True))
    
#clean the data, dont forget the soap 

train.replace(to_replace={'M','F','XNA'},value={1,2,3},inplace=True)
train.DEF_30_CNT_SOCIAL_CIRCLE.fillna(value=train['DEF_30_CNT_SOCIAL_CIRCLE'].mean(),inplace=True)
train.DEF_60_CNT_SOCIAL_CIRCLE.fillna(value=train['DEF_60_CNT_SOCIAL_CIRCLE'].mean(),inplace=True)
train.AMT_REQ_CREDIT_BUREAU_DAY.fillna(train['AMT_REQ_CREDIT_BUREAU_DAY'].mean(),inplace=True)
train.replace(to_replace={'Y','N'},value={0,1},inplace=True)
train.AMT_INCOME_TOTAL.fillna(value=168797.9192969845,inplace=True)
train.AMT_INCOME_TOTAL.fillna(value=168797.9192969845,inplace=True)
train.fillna(0,inplace=True)

#SET INPUTS
#make 
target = train.TARGET.values

features = train[['CODE_GENDER','AMT_REQ_CREDIT_BUREAU_DAY','NAME_EDUCATION_TYPE','NAME_CONTRACT_TYPE']].values

#clean the data, dont forget the soap 
from sklearn import model_selection
import sklearn 
from sklearn import linear_model
from sklearn import preprocessing
target = train.TARGET.values
features = train[['CODE_GENDER','AMT_REQ_CREDIT_BUREAU_DAY','NAME_EDUCATION_TYPE','NAME_CONTRACT_TYPE']].values
x_test,x_train,y_test,y_train = model_selection.train_test_split(features,target,test_size=.6,random_state=11)
classifier = linear_model.LogisticRegression()
classifier_ = classifier.fit(x_train,y_train)
cross_val_score = model_selection.cross_val_score
print(cross_val_score(classifier,x_test,y_test))
#SET INPUTS
#make inputs and targets np arrays with the .values function
target = train.TARGET.values
train.FLAG_EMP_PHONE
unscaled_inputs = train[['CODE_GENDER','AMT_REQ_CREDIT_BUREAU_DAY','NAME_EDUCATION_TYPE','NAME_CONTRACT_TYPE','FLAG_EMP_PHONE']].values
unscaled_targets = train['TARGET'].values

#balancing data half defaulted other half of samples did not default
one_targets = int(np.sum(unscaled_targets))
to_remove = []
zero_counter = 0
for i in range(unscaled_targets.shape[0]):
    if unscaled_targets[i]==0:
        zero_counter += 1
        if zero_counter > one_targets:
            to_remove.append(i)
unscaled_inputs_equal_priors = np.delete(unscaled_inputs,to_remove,axis=0)
targets_new = np.delete(unscaled_targets,to_remove,axis=0)
###standarize and shuffle data
from sklearn import preprocessing
inputs = preprocessing.scale(unscaled_inputs_equal_priors)
shuffled = np.arange(inputs.shape[0])
np.random.shuffle(shuffled)
shuffled_inputs = inputs[shuffled]
shuffled_targets = targets_new[shuffled]

sample_count = shuffled_inputs.shape[0]
train_samples_count = int(0.8*sample_count)
validation_samples_count = int(0.1*sample_count)
test_sample_count = sample_count - train_samples_count - validation_samples_count
#
train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]
#
validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]
#
test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]
np.savez
