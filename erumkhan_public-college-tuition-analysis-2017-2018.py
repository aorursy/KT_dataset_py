# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os
os.getcwd()
dirpath = os.getcwd()

data_dir = dirpath + '/MERGED2010_11_PP.csv/'
column_list = ["CONTROL","TUITIONFEE_IN"]

data_path = "/kaggle/input/college-tuition-data-20172018/MERGED2017_18_PP.csv"

df = pd.read_csv(data_path, usecols=column_list)

df
df["CONTROL"] = df["CONTROL"].astype(str)

df.info()
control_dict = {"1": "Public",

                "2": "Private nonprofit",

                "3": "Private for-profit"}

df["CONTROL"] = df["CONTROL"].map(control_dict)

df.info()
df.sample(10)
df = df[df["CONTROL"] != 'Private for-profit']

df.info()
df.sample(10)
df = df[df["CONTROL"] != 'Private nonprofit']

df.info()
df.sample(20)
df = df[df["TUITIONFEE_IN"] != 0]

df.info()
df = df[df["TUITIONFEE_IN"].isna() == False]

df.info()
# Random sample of 100 colleges

SAMPLE_SIZE = 100        



df_sample = df.sample(SAMPLE_SIZE)

sample_mean = df_sample["TUITIONFEE_IN"].mean()

sample_mean  
sample_std = df_sample["TUITIONFEE_IN"].std()

round(sample_std, 2)
std_err = sample_std / math.sqrt(SAMPLE_SIZE)       # standard error

std_err
LCL_68 = sample_mean -  std_err

UCL_68 = sample_mean +  std_err



print("Lower confidence limit at 68% confidence level = ", round(LCL_68,2))

print("Upper confidence limit at 68% confidence level = ", round(UCL_68,2))
LCL_95 = sample_mean -  2 * std_err

UCL_95 = sample_mean +  2 * std_err



print("Lower confidence limit at 95% confidence level = ", round(LCL_95,2))

print("Upper confidence limit at 95% confidence level = ", round(UCL_95,2))
LCL_997 = sample_mean -  3 * std_err

UCL_997 = sample_mean +  3 * std_err

print("Lower confidence limit at 99.7% confidence level = ", round(LCL_997,2))

print("Upper confidence limit at 99.7% confidence level = ", round(UCL_997,2))
df["TUITIONFEE_IN"].mean()
5640.53 - 5441.22