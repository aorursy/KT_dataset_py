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
from sklearn.ensemble import RandomForestRegressor

#Define some function to handle data
#######################################################
def get_shape_and_col_labels(df):
    #Get the shape of the data
    num_rows,num_cols = df.shape
    #Get the col labels 
    data_cols = df.columns.values.tolist()
    
    return num_rows,num_cols,data_cols
#######################################################

#######################################################
def get_result_out(df):
    #Get result field out and replace neg with 0 and pos with 1
    data_result = df.pop("class")
    data_result.replace('neg',0,inplace = True)
    data_result.replace('pos',1,inplace = True)

    return data_result
#######################################################

#Read in the Data Train and Test Set.
data = pd.read_csv("../input/aps_failure_training_set.csv")
data_test = pd.read_csv("../input/aps_failure_test_set.csv")

#Replace the 'na' values with NaN
data_result = get_result_out(data)
data_test_result = get_result_out(data_test)

data.replace('na',np.nan,inplace = True)
data_test.replace('na',np.nan,inplace = True)
#################################################################

#################################################################
#Eliminate some of the columns based on 2 heuristics

#Get the shape of the data
num_rows,num_cols,data_cols = get_shape_and_col_labels(data)

#Figure out the fraction of na values in the columns
num_nas = data.isnull().sum()
num_nas_vals = num_nas.values
num_nas_vals = num_nas_vals/num_rows
#print(num_nas_vals)

#HEURISTIC 1
#REMOVE Columns where the more than 10% of the entrys are 'na'
for col_index in range(num_cols):
    if num_nas_vals[col_index] > 0.1:
        x=data.pop(data_cols[col_index])
        x=data_test.pop(data_cols[col_index])
        
#Now that we have removed some of the columns
#Get the cols stuff again and the shape of the data
num_rows,num_cols,data_cols = get_shape_and_col_labels(data)


#Repalce NaN values with median
#Create a dictionary where the medians of each of the cols will be held
col_medians = {};
for col_label in data_cols:
    col_medians[col_label] = data[col_label].median()   
    
#Now Repalce NaN values with median
for col_label in data_cols:
    data[col_label].fillna(col_medians[col_label],inplace = True)
    data_test[col_label].fillna(col_medians[col_label],inplace = True)
    
for col_label in data_cols:
    if data[col_label].dtype == 'object':
        data[col_label] = data[col_label].astype(float)       
        data_test[col_label] = data_test[col_label].astype(float)       

#HEURISTIC 2 - Find rows where 90% of the entries are zero and
#and are zero even when the class is 'pos'
#For each column find the fraction of entries that are zero
frac_zeros = {}
for col_label in data_cols:
    frac_zeros[col_label] = (data[col_label] == 0.0).sum()/num_rows
    
#For the Columns that are largely zeros, find the correlation 
#between the non zero values and having a positive result
#Remove the columns where it is zero
sum_corr = {}
for col_label in data_cols:
    if frac_zeros[col_label] > 0.9:
        sum_corr[col_label] = (data[col_label] * data_result).sum()
        if sum_corr[col_label] == 0.0:
            x=data.pop(col_label)
            x=data_test.pop(col_label)

#Now use the Random forest Regressor from sklearn
model = RandomForestRegressor(n_estimators = 100,oob_score = True, random_state = 55)
model.fit(data,data_result)
rfc_prediction = model.predict(data_test) 

#Compute Cost of Prediction
Cost_1 = 10 #cost that an unnecessary check needs to be done by an mechanic at an workshop
Cost_2 = 500 # cost of missing a faulty truck, which may cause a breakdown
PredictionThreshold = 0.00

NumberOfCost_1 = (((rfc_prediction > PredictionThreshold) & (data_test_result == 0.0)).sum())
NumberOfCost_2 = (((rfc_prediction < PredictionThreshold) & (data_test_result == 1.0)).sum())

Total_Cost = Cost_1*NumberOfCost_1 + Cost_2*NumberOfCost_2

print('Total Cost: ', Total_Cost, NumberOfCost_1,  NumberOfCost_2)
