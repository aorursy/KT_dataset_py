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
# reading the training data
df_train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")


print(df_train.head)
#get info about the data
print(df_train.describe())
# spliting the Target value into the seperate columns
df_melted_test = df_train.pivot_table(index = ["Country_Region","Date"],columns = "Target",values="TargetValue")
print(df_melted_test.head())

#dropping of the two un-used columns and duplicate rows if any.
df_tem = df_train.drop(columns=["Target","TargetValue"])
df_train_tem = df_tem.drop_duplicates(subset = ["Country_Region","Date"])
print(df_train_tem.tail())
# merging the two dataframes to build into the single one.
df_train_final = pd.merge(df_train_tem,df_melted_test,on = ["Date","Country_Region"])
print(df_train_final)