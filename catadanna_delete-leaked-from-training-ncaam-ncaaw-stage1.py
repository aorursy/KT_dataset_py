import pandas as pd

import numpy as np



def concat_row(r):

    if r['WTeamID'] < r['LTeamID']:

        res = str(r['Season'])+"_"+str(r['WTeamID'])+"_"+str(r['LTeamID'])

    else:

        res = str(r['Season'])+"_"+str(r['LTeamID'])+"_"+str(r['WTeamID'])

    return res



# Delete leaked from train

def delete_leaked_from_df_train(df_train, df_test):

    df_train['Concats'] = df_train.apply(concat_row, axis=1)

    df_train_duplicates = df_train[df_train['Concats'].isin(df_test['ID'].unique())]

    df_train_idx = df_train_duplicates.index.values

    df_train = df_train.drop(df_train_idx)

    df_train = df_train.drop('Concats', axis=1)

    

    return df_train 



def read_data(inFile, sep=','):

    df_op = pd.read_csv(filepath_or_buffer=inFile, low_memory=False, encoding='utf-8', sep=sep)

    return df_op
PATH = "/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/"

df_test = read_data(PATH+"MSampleSubmissionStage1_2020.csv")

df_train = read_data(PATH+"MDataFiles_Stage1/MNCAATourneyCompactResults.csv")



print("SIZE TRAIN BEFORE :")

print(df_train.shape)

df_train = delete_leaked_from_df_train(df_train, df_test)

print()

print("SIZE TRAIN AFTER :")

print(df_train.shape)