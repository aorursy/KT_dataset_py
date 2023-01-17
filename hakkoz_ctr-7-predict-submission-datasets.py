# import packages

import csv

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import pickle

import gzip

import json
with open('../input/2020-digix-advertisement-ctr-prediction/test_data_B.csv', 'rb') as inputfile:

    test_B = pd.read_csv(inputfile, sep='|',index_col=0)
with open('../input/ctr-6-train-test-split-0/list_mapping_dict.json', 'rb') as inputfile2:

    list_mapping_dict = json.load(inputfile2)
with open('../input/ctr-6-split-sgd-batch-class-weight/sgd_model.pkl', 'rb') as inputfile3:

    sgd_model = pickle.load(inputfile3)
# fix mapping dict (convert keys to integer)

list_mapping_dict_fixed = []

for i, mapping_dict in enumerate(list_mapping_dict):

    if i==32:

        list_mapping_dict_fixed.append(mapping_dict)

        continue

    mapping_dict_fixed = {int(k):v for k,v in mapping_dict.items()}

    list_mapping_dict_fixed.append(mapping_dict_fixed)
testvalues = np.sort(test_B["task_id"].unique()).tolist()
map_values=list(list_mapping_dict_fixed[2].keys())
map_minus_test = [i for i in map_values if i not in testvalues]

len(map_minus_test)
test_minus_map = [i for i in testvalues if i not in map_values]

len(test_minus_map)
for i, col in enumerate(test_B.columns.tolist()):

    encoded=test_B[col].map(list_mapping_dict_fixed[i+1])

    downcasted = pd.to_numeric(encoded , downcast='float')

    test_B[col+'_tenc']=downcasted
test_B.info()
#drop original cols

test_B = test_B.drop(columns=[test_B.columns[i] for i in range(0,36)])

#drop pt_d_tenc         

test_B = test_B.drop(columns=["pt_d_tenc"])
test_B.isna().sum()
#fill na

for col_name in test_B.columns: 

    test_B[col_name].fillna(test_B[col_name].mean(), inplace=True)
test_B.isna().sum()
# predictions

y_pred = sgd_model.predict_proba(test_B)

submission_arr = y_pred[:,1].round(decimals=6)

id_arr = np.arange(1,len(submission_arr)+1)
# export submission

submissiondf=pd.DataFrame(zip(id_arr,submission_arr),columns=["id","probability"])

submissiondf.to_csv("submission.csv",index=False)