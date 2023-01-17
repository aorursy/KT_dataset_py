import numpy as np
import pandas as pd

# import the data

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

index = test_df.columns
index = list(index[1:])
X_train, y_train = train_df[index], train_df['Category']
def crime2num(str):
    crime_list = list(set(y_train.values))
    for i, item in enumerate(crime_list):
        if item == str:
            return i
my_dictionary = {k: f(v) for k, v in my_dictionary.items()}

my_dictionary = {k: v for k, v in my_dictionary.items()}