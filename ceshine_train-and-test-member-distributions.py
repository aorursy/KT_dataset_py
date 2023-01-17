# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from plotnine import * # TODO: avoid wildcard import



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
print('Loading data...')

data_path = '../input/'

train = pd.read_csv(data_path + 'train.csv')

test = pd.read_csv(data_path + 'test.csv')

members = pd.read_csv(data_path + 'members.csv', 

                      parse_dates=["registration_init_time", "expiration_date"])

members = members[members["expiration_date"] != "1970-01-01"].copy()
members_train = set(train["msno"].values)

members_test = set(test["msno"].values)

print("# of Members in train:\t", len(members_train))

print("# of Members in test:\t", len(members_test))
print("# of train only:\t", len(members_train - members_test))

print("# of test only: \t", len(members_test - members_train))

print("# of both:      \t", len(members_test & members_train))
members["scope"] = "both"

members.loc[members.msno.isin(members_train - members_test), "scope"] = "train"

members.loc[members.msno.isin(members_test - members_train), "scope"] = "test"
(

    ggplot(members) + 

    geom_histogram(aes(x="registration_init_time", fill="scope", y="..density.."), 

                   bins=100,  alpha = 0.2, position="identity") + 

    scale_fill_manual(values=("blue","red","green")) + theme_light() + 

    theme(figure_size=(8,4)) + 

    labs(x="Date", y="Norm Freq", title="Registration Date")

)
(

    ggplot(members) + 

    geom_histogram(aes(x="expiration_date", fill="scope", y="..density.."), 

                   bins=100,  alpha = 0.2, position="identity") + 

    scale_fill_manual(values=("blue","red","green")) + theme_light() + 

    theme(figure_size=(8,4)) + 

    labs(x="Date", y="Norm Freq", title="Expiration Date")

)
(

    ggplot(members[members.registration_init_time >= "2015-01-01"]) + 

    geom_histogram(aes(x="registration_init_time", fill="scope", y="..density.."), 

                   bins=100,  alpha = 0.2, position="identity") + 

    scale_fill_manual(values=("blue","red","green")) + theme_light() + 

    theme(figure_size=(8,4), axis_text_x=element_text(size=6)) + 

    labs(x="Date", y="Norm Freq", title="Registration Date")

)
(

    ggplot(members[

        (members.expiration_date >= "2015-07-01") & 

        (members.expiration_date < "2018-01-01")

    ]) + 

    geom_histogram(aes(x="expiration_date", fill="scope", y="..density.."), 

                   bins=100,  alpha = 0.2, position="identity") + 

    scale_fill_manual(values=("blue","red","green")) + theme_light() + 

    theme(figure_size=(8,4), axis_text_x=element_text(size=6)) + 

    labs(x="Date", y="Norm Freq", title="Expiration Date")

)