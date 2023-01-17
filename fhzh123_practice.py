import numpy as np

import pandas as pd

import time

import datetime

from collections import Counter



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
train = pd.read_csv("../input/train.tsv", delimiter = "\t")

test = pd.read_csv("../input/test.tsv", delimiter = "\t")
def data_seeing(dat_a):

    

    print("데이터 살펴보겠습니다.")

    print("")

    print("데이터의 총 개수는 {}개이며, 변수의 수는 {}개 입니다.".format(dat_a.shape[0], dat_a.shape[1]))

    print("해당 변수들은 {}입니다.".format(list(dat_a.columns)))

    print("")

    if dat_a.isnull().any().any():

        print("해당 데이터는 결측값이 존재합니다.")

        any_null = dat_a.isnull().any()

        print("결측값이 존재하는 변수의 수는 {}개 입니다.".format(sum(any_null)))

        na_list = list()

        for i in range(len(any_null.index)):

            if any_null[i]:

                na_list.append(any_null.index[i])

        print("결측값이 존재하는 변수는 {}입니다.".format(na_list))

    else:

        print("해당 데이터는 결측값이 존재하지 않습니다.")
data_seeing(train)
data_seeing(test)
def submit_csv_make(test_dat, pred):

    

    global submission

    

    submission = pd.DataFrame()

    

    submission["test_id"] = test.test_id

    submission["price"] = pred

    

    submission.to_csv("submission.csv", index = False)
submit_csv_make(test, pred_test)