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



trainfile="../input/train.tsv"

testfile="../input/test.tsv"



dataset=pd.read_csv(trainfile,delimiter="\t")

test_pre=pd.read_csv(testfile,delimiter="\t")



test_pre["price"]=dataset["price"].median()

test_pre.index=test_pre["test_id"]

test_pre.set_index("test_id",inplace=True)



test_pre[["price"]].to_csv("sub_median.csv",index_label=["test_id"])


