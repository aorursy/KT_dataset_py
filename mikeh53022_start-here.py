# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def build_answer_set(array_of_answers):
    # replace " with '
    # wrap the entire answer with "
    # remove all internal white space
    out = []
    for idx, ans in enumerate(array_of_answers, start=1):
        s = str(ans)
        s = s.replace(' ', '')
        val = {"Id":idx, "Answer": s}
        out.append(val)
        
    return out
# start your code here
ans1 = 3
ans2 = [10,12,13]
ans3 = {'Tom':  23 }

data = build_answer_set([ans1, ans2, ans3])


df = pd.DataFrame(data).set_index('Id')
print(df.head())

df.to_csv("submit.csv")

