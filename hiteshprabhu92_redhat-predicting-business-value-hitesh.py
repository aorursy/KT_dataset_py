# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# Reading in files:

act_test = pd.read_csv("../input/act_test.csv")

act_train = pd.read_csv("../input/act_test.csv")

people = pd.read_csv("../input/people.csv")



def feature_summary(data):

    n_row = data.shape[0]

    features = pd.DataFrame()

    features_names = []

    features_counts = []

    features_missing = []

    names = data.columns.values

    for i in names:

        features_names.append(i)

        features_counts.append(data[i].value_counts().count())

        features_missing.append(data[data[i].isnull()].shape[0])

    features['name'] = features_names

    features['value counts'] = features_counts

    features['missing'] = features_missing

    features['percentage_missing'] = features['missing']/n_row

    return (features)



feature_summary(act_test)
feature_summary(act_train)
feature_summary(people)
