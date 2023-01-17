# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train_ver2.csv",nrows=1000)

df.head()
unique_ids   = pd.Series(df["ncodpers"].unique())
unique_ids
reader = pd.read_csv('../input/train_ver2.csv', iterator=True)

loop = True

loop_count = 0

while loop:

    try:

        df = reader.get_chunk(10000)

        loop_count += 1

    except StopIteration:        

        loop = False

print(loop_count)

        

#正式开始愉快地玩数据。模仿从7000000个数据中取样10000个

reader = pd.read_csv('../input/train_ver2.csv', iterator=True)

df = reader.get_chunk(7000000)

#unique_ids   = pd.Series(df["ncodpers"].unique())

unique_ids = pd.Series(df['ncodpers'].unique()) #构造一个Series保存所有出现过的用户编码

limit_people = 1e4  #从中采样10000个用户

#unique_id    = unique_ids.sample(n=limit_people)

unique_id = unique_ids.sample(n=limit_people)  #sample函数的使用

#df           = df[df.ncodpers.isin(unique_id)]

df = df[df.ncodpers.isin(unique_id)]   #条件过滤

df.describe()