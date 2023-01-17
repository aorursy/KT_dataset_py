!pip install https://s3.amazonaws.com/h2o-release/datatable/stable/datatable-0.8.0/datatable-0.8.0-cp36-cp36m-linux_x86_64.whl
import time

import numpy as np

import pandas as pd

import datatable as dt

print(dt.__version__)
## Data Table Reading

start = time.time()

dt_df = dt.fread("../input/loan.csv")

end = time.time()

print(end - start)
import time



start = time.time()

pd_df= pd.read_csv("../input/loan.csv")

end = time.time()

print(end - start)
start = time.time()

dt_df.to_pandas()

end = time.time()

print(end - start)
dt_df.head()
# number of rows and columns

dt_df.shape
# To get the column names

dt_df.names[:10]
dt_df.mean()
## pd_df.mean()
start = time.time()

dt_df.sort("loan_amnt")

end = time.time()

print(end - start)
start = time.time()

pd_df.sort_values(by="loan_amnt")

end = time.time()

print(end - start)
start = time.time()

for i in range(100):

    dt_df[:, dt.sum(dt.f.loan_amnt), dt.by(dt.f.grade)]

end = time.time()

print(end - start)
start = time.time()

for i in range(100):

    pd_df.groupby("grade")["loan_amnt"].sum()

end = time.time()

print(end - start)
start = time.time()

for i in range(100):

    dt_df[dt.f.loan_amnt>dt.mean(dt.f.loan_amnt), "loan_amnt"]

end = time.time()

print(end - start)
start = time.time()

for i in range(100):

    pd_df["loan_amnt"][pd_df["loan_amnt"] > pd_df["loan_amnt"].mean()]

end = time.time()

print(end - start)
from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(dt_df[:,["loan_amnt", "installment"]], dt_df[:,"int_rate"])

model.coef_
gdf = dt_df[:, dt.sum(dt.f.loan_amnt), dt.by(dt.f.grade)]

gdf.to_csv("temp.csv")