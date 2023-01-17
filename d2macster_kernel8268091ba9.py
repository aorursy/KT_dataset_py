import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import os
df = pd.read_csv("../input/PS_20174392719_1491204439457_log.csv") # loading transactions
df.loc[0:10] # prining a sample of transactions
df_fraud = df[df["isFraud"] == 1] # number of fradulent transactions
isFraudCount = df_fraud.shape[0]
totalCount = df.shape[0] # total number of transactions
print("Fraud rate: {}, total transactions {}".format(isFraudCount / totalCount, totalCount)) # stats on fraud
print("Fraud transaction types are {}".format(df_fraud["type"].unique())) # which transactio types are fradulent. ?
df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
df = df.reset_index(drop=True).drop(columns=["isFlaggedFraud"])
df["origLetter"] = df["nameOrig"].map(lambda x: x[0])
df["destLetter"] = df["nameDest"].map(lambda x: x[0])
print(df[df["origLetter"] == 'M'].shape) # there are no transactions involing merchant
print(df[df["destLetter"] == 'M'].shape)
df = df.drop(columns=["origLetter", "destLetter"])
