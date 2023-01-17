#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Tue Jun 16 17:20:25 2020



@author: kurtschuepfer1

"""

import pickle

import pandas as pd

###########

# Read in pred dicts

# 

        

with open('../input/predictionsfinal/new_rf_pred_dict.pkl', 'rb') as file:

    rf_pred_dict = pickle.load(file)



rf_output_dict = dict()

for k, v in rf_pred_dict.items():    

    Input = rf_pred_dict[k]

    Output = list(filter(lambda x:1 in x, Input)) 

    # Find all posiitve labels in list of tuples. 

    Output = [item for item in Input 

              if item[1] in [1, 3, 4, 6, 7]] 

    rf_output_dict[k] = Output







# Pull excerpts with positive labels

rf_final_paperids = []

rf_final_excerpts = []

rf_final_labels = []



for k,v in rf_output_dict.items():

    if len(v) != 0:

        for elem in v:

            rf_final_paperids.append(k)

            rf_final_excerpts.append(elem[0])

            rf_final_labels.append(elem[1])



# Create final output data frame

rf_final_df = pd.DataFrame(

    {'paperid': rf_final_paperids,

     'excerpt': rf_final_excerpts,

     'label': rf_final_labels

    })

ann_output_dict = dict()

with open('../input/predictionsfinal/new_ann_pred_dict.pkl', 'rb') as file:

    new_ann_pred_dict = pickle.load(file)

    

for k, v in new_ann_pred_dict.items():    

    Input = new_ann_pred_dict[k]

    Output = list(filter(lambda x:1 in x, Input)) 

    # Find all posiitve labels in list of tuples. 

    Output = [item for item in Input 

              if item[1] in [0, 2, 4, 6, 8]] 

    ann_output_dict[k] = Output





# Pull excerpts with positive labels

ann_final_paperids = []

ann_final_excerpts = []

ann_final_labels = []



for k,v in ann_output_dict.items():

    if len(v) != 0:

        for elem in v:

            ann_final_paperids.append(k)

            ann_final_excerpts.append(elem[0])

            ann_final_labels.append(elem[1])



# Create final output data frame

ann_final_df = pd.DataFrame(

    {'paperid': ann_final_paperids,

     'excerpt': ann_final_excerpts,

     'label': ann_final_labels

    })





softmax_map = {0: 1, 

               1: 10, 

               2: 3, 

               3:30, 

               4: 4, 

               5: 40, 

               6: 6, 

               7: 60, 

               8: 7, 

               9:70}



ann_final_df['label'] = ann_final_df['label'].map(softmax_map)







# RF inclusive

with open('../input/predictionsfinal/new_rf_pred_dict.pkl', 'rb') as file:

    rf_pred_dict = pickle.load(file)



rf_output_dict = dict()

for k, v in rf_pred_dict.items():    

    Input = rf_pred_dict[k]

    Output = list(filter(lambda x:1 in x, Input)) 

    Output = [item for item in Input 

              if item[1] in [1, 10, 3, 30, 4, 40,  6, 60,  7, 70]] 

    # Find all posiitve labels in list of tuples. 

    rf_output_dict[k] = Output



# Pull excerpts with positive labels

rf_final_paperids = []

rf_final_excerpts = []

rf_final_labels = []



for k,v in rf_output_dict.items():

    if len(v) != 0:

        for elem in v:

            rf_final_paperids.append(k)

            rf_final_excerpts.append(elem[0])

            rf_final_labels.append(elem[1])



# Create final output data frame

rf_inclusive_df = pd.DataFrame(

    {'paperid': rf_final_paperids,

     'excerpt': rf_final_excerpts,

     'label': rf_final_labels

    })





rf_map = {10: 1, 

               30: 3, 

               40: 4, 

               60:6, 

               70: 7}



rf_inclusive_df['label'] = rf_inclusive_df['label'].map(rf_map)



# Write results

rf_final_df.to_csv('rf_final_df.csv')

ann_final_df.to_csv('ann_final_df.csv')

rf_inclusive_df.to_csv('rf_inclusive_df.csv')