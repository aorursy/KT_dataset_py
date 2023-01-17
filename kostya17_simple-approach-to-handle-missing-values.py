import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



train_df = pd.read_csv('../input/train.csv', na_values="-1")

test_df = pd.read_csv('../input/test.csv', na_values="-1")
def describe_missing_values(df):

    na_percent = {}

    N = df.shape[0]

    for column in df:

        na_percent[column] = df[column].isnull().sum() * 100 / N



    na_percent = dict(filter(lambda x: x[1] != 0, na_percent.items()))

    plt.bar(range(len(na_percent)), na_percent.values())

    plt.ylabel('Percent')

    plt.xticks(range(len(na_percent)), na_percent.keys(), rotation='vertical')

    plt.show()
print("Missing values for Train dataset")

describe_missing_values(train_df)



print("Missing values for Test dataset")

describe_missing_values(test_df)
target = train_df.target

test_id = test_df.id

train_df.drop(["id", "target", "ps_car_03_cat", "ps_car_05_cat"], axis=1, inplace=True)

test_df.drop(["id", "ps_car_03_cat","ps_car_05_cat"], axis=1, inplace=True)
cat_cols = [col for col in train_df.columns if 'cat' in col]

bin_cols = [col for col in train_df.columns if 'bin' in col]

con_cols = [col for col in train_df.columns if col not in bin_cols + cat_cols]



for col in cat_cols:

    train_df[col].fillna(value=train_df[col].mode()[0], inplace=True)

    test_df[col].fillna(value=test_df[col].mode()[0], inplace=True)

    

for col in bin_cols:

    train_df[col].fillna(value=train_df[col].mode()[0], inplace=True)

    test_df[col].fillna(value=test_df[col].mode()[0], inplace=True)

    

for col in con_cols:

    train_df[col].fillna(value=train_df[col].mean(), inplace=True)

    test_df[col].fillna(value=test_df[col].mean(), inplace=True)
print("Missing values for Train dataset")

describe_missing_values(train_df)



print("Missing values for Test dataset")

describe_missing_values(test_df)