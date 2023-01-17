import pandas as pd

import numpy as np

import os

import pickle

import gc

import xgboost as xgb

import numpy as np

import re

import pandas as pd

from sklearn.model_selection import train_test_split



samplePath = "../input/" + "en_sample_submission.csv"

testPath = "../input/" + "en_test.csv"

trainPath = "../input/" + "en_train.csv"



df_sample = pd.read_csv(samplePath)

df_test = pd.read_csv(testPath)

df_train = pd.read_csv(trainPath)



# Any results you write to the current directory are saved as output.
# Which words were different?

df_train_nochange = df_train[df_train["before"] == df_train["after"]]

df_train_change = df_train[df_train["before"] != df_train["after"]]

print("Train head: ", df_train_change.head())



# About how many changed in before and after?

totalcount = df_train.size

changecount = df_train[df_train["before"] != df_train["after"]].size

print("Dataset changers were about: ", (changecount/totalcount)*100, "%")



#Which class labels didn't change ever?

#Which class labels always changed?

#Which class labels flip flopped?

staticClasses = set(df_train[df_train["before"] == df_train["after"]]["class"].unique())

dynamicClasses = set(df_train[df_train["before"] != df_train["after"]]["class"].unique())

neverchanged = set([i for i in staticClasses if i not in dynamicClasses])

alwayschanged = set([i for i in dynamicClasses if i not in staticClasses])

sometimeschanged = set([i for i in staticClasses if i in dynamicClasses and i in staticClasses])

#print("Static classes: ", staticClasses)

#print("Dynamic classes: ", dynamicClasses)

print("These classes never changed: \n\t", neverchanged)

print("These classes always changed: \n\t", alwayschanged)

print ("These classes sometimes changed: \n\t", sometimeschanged)



# Confirm the above results were true, indeed, it makes sense.

print("never changers IN df_train_change? ", neverchanged in list(df_train_change["class"].unique()))

print("always changers IN df_train_nochange? ", alwayschanged in list(df_train_nochange["class"].unique()))
# Class Label

y_data =  pd.factorize(df_train['class'])

df_train["class_num"] = y_data[0]
# Create an array of characters by asci code



count = 0

maxlen = max(df_train["before"].apply(lambda x:len(str(x))))

x_data = []

for x in df_train["before"].values:

    char_range = len(list(str(x)))

    x_row = np.zeros(char_range, dtype=int)

    for i in range(char_range):

        c = ord(str(x)[i])

        x_row[i] = c

    

    

    if(count < 5):

        count += 1

        print(x_row)
