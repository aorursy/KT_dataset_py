import numpy as np
import pandas as pd

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/train.csv")
sample_sub = pd.read_csv("../input/sample_submission.csv")


print("______Training Data______")
print(train_data)
train_data.info()

print("______Test Data______")
print(test_data)
test_data.info()


print("_____Train Data_____")
train_nulls = train_data.isnull().sum()
#print(train_nulls)
#print("*"*40)
#print("_____Test Data_____")
test_nulls = test_data.isnull().sum()
print(test_nulls)
train_nulls.where(train_nulls != test_nulls)
train_data.dropna(axis=1, thresh=584) # drop features (columns) that have at least 584 null values (40%)
train_data.dropna(axis=1, how='any')
train_nonull = train_data.dropna(axis=1, thresh=584)
test_nonull = test_data.dropna(axis=1, thresh=584)