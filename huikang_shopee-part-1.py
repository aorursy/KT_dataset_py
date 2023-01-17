# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from IPython.display import display, HTML

def displayer(df): display(HTML(df.head(2).to_html()))

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)
df1 = pd.read_csv("/kaggle/input/undrg-rd1-listings/Extra Material 2 - keyword list_with substring.csv")

df2 = pd.read_csv("/kaggle/input/undrg-rd1-listings/Keyword_spam_question.csv")
from collections import defaultdict



dic = defaultdict(list)

for group, keywords in zip(list(df1["Group"])[::-1], list(df1["Keywords"])[::-1]):

    for keyword in keywords.replace(", ", ",").split(","):

        dic[group].append(keyword)

    

dic2 = {}

for key in dic:

    for ls in dic[key]:

        dic2[ls] = key
# dic3 = {}

# for key1 in dic2:

#     for key2 in dic2:

#         dic3[key2] = []

#         if key2 in key1 and key2 != key1:

#             dic3[key2].append(key1)
dic3["printer toner"]
names = list(df2["name"])

names = [re.sub(r'[^\x00-\x7f]',r'', n) for n in names]

PERMITTED_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ " 

names = [n.lower() for n in names]

names = ["".join(c for c in n if c in PERMITTED_CHARS) for n in names]
names
res = []

for name in names[:]:

    lst = []

    for key in dic2:

        if key in name:

            lst.append(dic2[key])

    res.append(lst)

    

res = [r[::-1] for r in res]
df_res = pd.DataFrame({"index": range(len(res)),

              "groups_found": res})

df_res.to_csv("submission.csv", index = False)
pd.read_csv("submission.csv")