import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input"))

import sys 

sys.path.append ('../input/montecarlomodelselection-functions/')
from MonteCarloModelSelection_Functions import *      


%matplotlib inline
%autosave 0
# Loading dataset creditcard
filename = '../input/creditcardfraud/creditcard.csv'   

with open(filename, 'r') as f:
    reader=csv.reader(f, delimiter=',') 
    labels=next(reader)

    raw_data=[]
    for row in reader:
        raw_data.append(row)

data = np.array(raw_data)
data = data.astype(np.float)
# Setting target and data
target = data[:,-1]
dataAmount   = data[:,29]
data   = data[:,1:29]

# Normalising Amount column 
dataAmountNormalize = np.array((dataAmount-np.mean(dataAmount))/np.std(dataAmount))
data = np.c_[ data,dataAmountNormalize]
# Output Path
path = './output/'
# Calculating transformed dataset by means of logit or normal method
transformation = 'logit' 
transformed_dataset = Transformation(data, target, transformation)
# Calculating all metric
metric ='all'
global_pi = Calculate_Metrics(transformed_dataset, target, metric, path, transformation)
# Calculating new datasets with combinations of products of features using distance metric
threshold = 0.6
transformation = 'logit'
metric = 'all'
metric_prod = 'distance'
new_dataset, new_dataset_df = Products_Analysis(data, transformed_dataset, target, global_pi, metric, metric_prod, transformation, path, threshold)
# Calculating new datasets with combinations of products of features using roc metric
threshold = 0.6
transformation = 'logit'
metric = 'all'
metric_prod = 'roc'
new_dataset, new_dataset_df = Products_Analysis(data, transformed_dataset, target, global_pi, metric, metric_prod, transformation, path, threshold)
new_dataset_df.tail(20)
X_train, X_test, y_train, y_test = train_test_split(new_dataset, target, test_size = 0.3, random_state = 0)
# Resampling dataset  
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
np.random.seed(10)
number_records_fraud = target.sum().astype(int)
normal_indices = (target==0).nonzero()[0]
fraud_indices = (target==1).nonzero()[0]
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
under_sample_data = new_dataset[under_sample_indices,:]
X_undersample = under_sample_data
y_undersample = target[under_sample_indices]
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample, y_undersample, test_size = 0.3, random_state = 0)
metric = 'Distance'
number_iterations = 10000
number_ini_ratio = 5
number_final_ratio = 5
results= Multivariate_Best_Model(number_iterations, X_train_undersample, y_train_undersample, X_test_undersample, y_test_undersample, metric, path, number_ini_ratio, number_final_ratio)       
results.head(5)
models_list = [i-1 for i in results['Models'][0]]
bt = results['Betas'][0]
ind_best = models_list 
X_test_b = X_test_undersample[:,ind_best]
X_test_b_1 = np.array([1]*X_test_b.shape[0])
X_test_b_ = np.c_[X_test_b_1, X_test_b]
xtest_bt = np.ravel(np.dot(X_test_b_,np.transpose(bt)))

[tn_u, fp_u, fn_u, tp_u] = Graph(y_test_undersample, xtest_bt)
models_list = [i-1 for i in results['Models'][0]]
bt = results['Betas'][0]
ind_best = models_list 
X_test_b = X_test[:,ind_best]
X_test_b_1 = np.array([1]*X_test_b.shape[0])
X_test_b_ = np.c_[X_test_b_1, X_test_b]
xtest_bt = np.ravel(np.dot(X_test_b_,np.transpose(bt)))

[tn, fp, fn, tp]  = Graph(y_test, xtest_bt)
