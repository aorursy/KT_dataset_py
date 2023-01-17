import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv("../input/act_train.csv")
train.head()
#how many columns and rows are there? 
train.shape 
#data types 
train.dtypes
#some stats 
train.describe()
pd.value_counts(train["outcome"].values.ravel())
#submission file format 
submission_format= pd.read_csv("../input/sample_submission.csv")
submission_format.head()
#for each column what is the distribution of unique values ? 

for index,col in enumerate(list(train.columns.values)):
    print("unique values for column "+col)
    print(pd.value_counts(train[col].values.ravel())) 
#number of unique values for each column 
for index,col in enumerate(list(train.columns.values)):
    print("unique values for column "+col)
    print(len(pd.unique(train[col].values.ravel()))) 
# a peak into the people file

people = pd.read_csv("../input/people.csv")
people.head() 
#new_train = train[["activity_id","date","people_id","outcome"]]
#train = pd.get_dummies(train[train.columns.difference(["activity_id","date","people_id","outcome"])])
#train.head()

test = pd.read_csv("../input/act_test.csv")
test.head() 
