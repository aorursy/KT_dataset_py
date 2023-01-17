# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
file_path = "/kaggle/input/m5-forecasting-accuracy/"
path_eval = "sales_train_evaluation.csv"
path_val = "sales_train_validation.csv"
path_price = "sell_prices.csv"
path_calender = "calendar.csv"

train_eval = pd.read_csv(file_path + path_eval)
train_val = pd.read_csv(file_path + path_val)
sell_prices = pd.read_csv(file_path + path_price)
calender = pd.read_csv(file_path + path_calender)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
print(train_eval.head().append(train_eval.tail()), "\n")
print("Shape:", train_eval.shape, "\n")
print("Missing values: \n", train_eval.isna().sum().sort_values(ascending=False), "\n")
print("Data types:\n", train_eval.iloc[:,:7].dtypes)
print(train_val.head().append(train_val.tail()), "\n")
print("Shape:", train_val.shape, "\n")
print("Missing values: \n", train_val.isna().sum().sort_values(ascending=False), "\n")
print("Data types:\n", train_val.iloc[:,:7].dtypes)
print(sell_prices.head().append(sell_prices.tail()), "\n")
print("Shape:", sell_prices.shape, "\n")
print("Missing values: \n", sell_prices.isna().sum().sort_values(ascending=False), "\n")
print("Data types:\n", sell_prices.dtypes)
print(calender.head().append(calender.tail()), "\n")
print("Shape:", calender.shape, "\n")
print("Missing values: \n", calender.isna().sum().sort_values(ascending=False), "\n")
print("Data types:\n", calender.dtypes)