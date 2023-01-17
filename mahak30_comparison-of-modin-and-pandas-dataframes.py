!pip install modin[dask]

!pip install modin[ray]

!pip install -U ipykernel



import time#creates a time comparison report for modin and pandas

import numpy as np

import pandas as pd

import modin.pandas as md

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
s = time.time()

df_pandas = pd.read_csv('/kaggle/input/titanic/train.csv')

e = time.time()

print("Pandas Loading Time = {}".format(e-s))
s = time.time()

df_modin = md.read_csv("/kaggle/input/titanic/train.csv")

e = time.time()

print("Modin Loading Time = {}".format(e-s))
s = time.time()

df_pandas_concat = pd.concat([df_pandas for _ in range(5)])

e = time.time()

print("Pandas Concat Time = {}".format(e-s))
s = time.time()

df_modin_concat = md.concat([df_modin for _ in range(5)])

e = time.time()

print("Modin Concat Time = {}".format(e-s))
s = time.time()

t_pandas = df_pandas_concat.isnull()

e = time.time()

print("Pandas Null Time Check = {}".format(e-s))
s = time.time()

t_modin = df_modin_concat.isnull()

e = time.time()

print("Modin Null Time Check = {}".format(e-s))
s = time.time()

max_pandas = df_pandas_concat["Fare"].max()

e = time.time()

print ("Max value in Pandas Dataframe = {}".format(max_pandas))

print("Pandas Count Time = {}".format(e-s))
s = time.time()

max_modin = df_modin_concat["Fare"].max()

e = time.time()

print ("Max value in Modin Dataframe = {}".format(max_modin))

print("Modin Count Time = {}".format(e-s))