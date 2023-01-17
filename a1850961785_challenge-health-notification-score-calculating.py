import numpy as np

import pandas as pd

import joblib

import lightgbm as lgb

import time

import pickle

import math

import string

import datetime

from sklearn.model_selection import KFold

from sklearn.metrics import f1_score, mean_squared_error

import glob

import json



def challenge_data_conversion(challenge_data):

    output = []

    output.append(challenge_data['id'])

    output.append(1 if len(challenge_data['winners']) > 0 else 0)

    output.append(len(challenge_data['winners']))

    

    return output



def data_conversion(training_file_path):

    data_df = pd.DataFrame(columns=['id', 'hasWinner', 'numOfWinners'])

    file_list = []

    extensions = ["json"]

    for extension in extensions:

        file_glob = glob.glob(training_file_path+"/*."+extension)

        file_list.extend(file_glob)

    print(str(len(file_list))+' files')

        

    for file_path in file_list:

        with open(file_path,'r') as f:

            data_dict = json.load(f)

        for challenge_data in data_dict:

            #try:

            data_df.loc[len(data_df)] = challenge_data_conversion(challenge_data)

            #except:

            #    print(challenge_data_conversion(challenge_data))

            

            

    return data_df



test_data = data_conversion('../input/challenge-health-notification-test-data/')

reg_output = pd.read_csv('../input/challenge-health-notification-reg-output/lightgbm_numOfWinners_prediction.csv')

cls_output = pd.read_csv('../input/challenge-health-notification-cls-output/lightgbm_hasWinner_prediction.csv')
print('f1 score:')

print(f1_score(test_data['hasWinner'].astype("int").values, cls_output['hasWinner'].values))



print('')



print('mean squared error:')

print(mean_squared_error(test_data['numOfWinners'].values, reg_output['numOfWinners'].values))