import pandas as pd
import dask.dataframe as dd

# confirming the default pandas doesn't work (running the below code should result in a memory error)
# data = pd.read_csv("../input/riiid-test-answer-prediction/train.csv")
%%time

dtypes = {
    "row_id": "int64",
    "timestamp": "int64",
    "user_id": "int32",
    "content_id": "int16",
    "content_type_id": "boolean",
    "task_container_id": "int16",
    "user_answer": "int8",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32", 
    "prior_question_had_explanation": "boolean"
}

data = pd.read_csv("../input/riiid-test-answer-prediction/train.csv", dtype=dtypes)

print("Train size:", data.shape)
data.head()
%%time

dtypes = {
    "row_id": "int64",
    "timestamp": "int64",
    "user_id": "int32",
    "content_id": "int16",
    "content_type_id": "boolean",
    "task_container_id": "int16",
    "user_answer": "int8",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32", 
    "prior_question_had_explanation": "boolean"
}

data = dd.read_csv("../input/riiid-test-answer-prediction/train.csv", dtype=dtypes).compute()

print("Train size:", data.shape)
data.head()
# datatable installation with internet
# !pip install datatable==0.11.0 > /dev/null

# datatable installation without internet
!pip install ../input/python-datatable/datatable-0.11.0-cp37-cp37m-manylinux2010_x86_64.whl > /dev/null

import datatable as dt
%%time

data = dt.fread("../input/riiid-test-answer-prediction/train.csv")

print("Train size:", data.shape)
data.head()
# rapids installation (make sure to turn on GPU)
import sys
!cp ../input/rapids/rapids.0.15.0 /opt/conda/envs/rapids.tar.gz
!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null
sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path

import cudf
%%time

data = cudf.read_csv("../input/riiid-test-answer-prediction/train.csv")

print("Train size:", data.shape)
data.head()
# reading data from csv using datatable and converting to pandas
# data = dt.fread("../input/riiid-test-answer-prediction/train.csv").to_pandas()

# writing dataset as csv
# data.to_csv("riiid_train.csv", index=False)

# writing dataset as hdf5
# data.to_hdf("riiid_train.h5", "riiid_train")

# writing dataset as feather
# data.to_feather("riiid_train.feather")

# writing dataset as parquet
# data.to_parquet("riiid_train.parquet")

# writing dataset as pickle
# data.to_pickle("riiid_train.pkl.gzip")

# writing dataset as jay
# dt.Frame(data).to_jay("riiid_train.jay")
%%time

dtypes = {
    "row_id": "int64",
    "timestamp": "int64",
    "user_id": "int32",
    "content_id": "int16",
    "content_type_id": "boolean",
    "task_container_id": "int16",
    "user_answer": "int8",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32", 
    "prior_question_had_explanation": "boolean"
}

data = pd.read_csv("../input/riiid-test-answer-prediction/train.csv", dtype=dtypes)

print("Train size:", data.shape)
%%time

data = pd.read_feather("../input/riiid-train-data-multiple-formats/riiid_train.feather")

print("Train size:", data.shape)
%%time

data = pd.read_hdf("../input/riiid-train-data-multiple-formats/riiid_train.h5", "riiid_train")

print("Train size:", data.shape)
%%time

data = dt.fread("../input/riiid-train-data-multiple-formats/riiid_train.jay")

print("Train size:", data.shape)
%%time

data = pd.read_parquet("../input/riiid-train-data-multiple-formats/riiid_train.parquet")

print("Train size:", data.shape)
%%time

data = pd.read_pickle("../input/riiid-train-data-multiple-formats/riiid_train.pkl.gzip")

print("Train size:", data.shape)