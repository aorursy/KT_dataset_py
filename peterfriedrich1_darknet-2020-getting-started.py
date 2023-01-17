import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from pandas.api.types import is_string_dtype, is_numeric_dtype

from sklearn.model_selection import train_test_split



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import forest



import math

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
parse_dates = ["Timestamp"]
# will need to skip line 328

df_raw = pd.read_csv("/kaggle/input/cicdarknet2020-internet-traffic/Darknet.CSV", parse_dates=parse_dates, error_bad_lines=False)
df_raw.head()
df_raw.shape
def display_all(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(df)

display_all(df_raw.tail().T)