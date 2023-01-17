# installation with internet

# !pip install datatable==0.11.0
# installation without internet

!pip install ../input/python-datatable/datatable-0.11.0-cp37-cp37m-manylinux2010_x86_64.whl
%%time



# reading the dataset from raw csv file

import datatable as dt



train = dt.fread("../input/riiid-test-answer-prediction/train.csv").to_pandas()



print(train.shape)

train.head()
# saving the dataset in .jay (binary format)

dt.fread("../input/riiid-test-answer-prediction/train.csv").to_jay("train.jay")
%%time



# reading the dataset from .jay format

import datatable as dt



train = dt.fread("train.jay")



print(train.shape)
train
%%time



import datatable as dt



train = dt.fread("train.jay").to_pandas()



print(train.shape)

train.head()