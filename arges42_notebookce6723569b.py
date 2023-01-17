import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_json("../input/train.json")

test = pd.read_json("../input/test.json")
train.head()
print("Train shape: ",train.shape)

print("Test shape: ",test.shape)
print("Variable types:")

print(train.dtypes)
print("Empty fields in train:")

print(train.isnull().sum())

print("Empty fields in test:")

print(test.isnull().sum())
train.bathrooms.plot.bar()
train.bedr