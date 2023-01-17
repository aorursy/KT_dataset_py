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
df = pd.read_csv("../input/train.csv")

df = pd.read_csv("../input/submission.csv")

df.to_csv("submission_new.csv", index=False)
df
from collections import Counter

import re



# df['land'] = df['交易筆棟數'].apply(lambda x: re.findall("土地\d{1,2}", x))

# df[['交易筆棟數', 'land']]

df['備註']