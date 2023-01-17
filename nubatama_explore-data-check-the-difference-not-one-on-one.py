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
# Read en_train.csv  file.

train_df = pd.read_csv(filepath_or_buffer="../input/en_train.csv", encoding="utf-8", dtype={'class':'category'})

train_df
train_df.info()
train_df["class"].cat.categories
# missing  data check

isnull_num  = train_df.isnull().sum(axis = 0)

isnull_num
train_df["diff"] = train_df["before"] != train_df["after"]

train_df
total_num = len(train_df)

diff_num = train_df["diff"].sum()

print("total {0} / diff {1} / rate {2}".format(total_num, diff_num, diff_num / total_num))
groupby = train_df[["class", "diff"]].groupby("class").agg([np.sum, np.size]).reset_index()

groupby
groupby.plot(kind="bar")
diff_df = train_df[train_df["diff"]]

diff_df = diff_df.loc[:,"class":"after"].drop_duplicates()

diff_df = diff_df.sort_values(by=["class","before"])

diff_df
group_df = diff_df.groupby(["class","before"])

group_df.filter(lambda x: len(x)>1)
train_df.iloc[2265:2275]
train_df.iloc[660762:660780]
train_df.iloc[7665:7675]
train_df.iloc[322660:322670]
train_df2 = train_df.loc[:,"class":"after"].drop_duplicates()

train_df2 = train_df.sort_values(by=["class","before"])

train_df_gr = train_df.groupby(["class","before"])

train_df_gr.filter(lambda x: len(x)>1)
## Check test and sample  submission data
test_df = pd.read_csv(filepath_or_buffer="../input/en_test.csv", encoding="utf-8" )

test_df
sample_df = pd.read_csv(filepath_or_buffer="../input/en_sample_submission.csv", encoding="utf-8" )

sample_df