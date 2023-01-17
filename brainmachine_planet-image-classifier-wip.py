# IMPORTS



import numpy as np

import pandas as pd





# print list of data files

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# lets look at the sample_submission.csv file...

sample_csv = pd.read_csv("../input/sample_submission.csv")

sample_csv.head()
# lets look at the training data

train = pd.read_csv("../input/train.csv")

train.shape
# looks like 40478 rows (with the HEADER) and two columns. let's verify

train.head()

print(len(train))

# Jesse: We don't count the header

# lets look at the tags

tags = train['tags'].apply(lambda x: x.split(' '))

tags.head(n=20)
# lets count the instances of them

# TODO

tags.groupby(['tags'])