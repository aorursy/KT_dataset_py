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
import pandas as pd

df = pd.DataFrame()

mapping = {}

source_path = "../input/aws dataset/aws dataset"

df2 = pd.read_csv("../input/aws-dataset/aws dataset/aws dataset/output-onlinetsvtools_2.csv")
df2.columns
df2.effectiveness
df['label'] = df2.effectiveness

df['review'] = df2.benefitsReview
df = df.sample(frac=1).reset_index(drop=True)
# df.to_csv('/home/pralok/Desktop/train.csv', index=False, header=False)
df.head()
df_test = pd.read_csv("../input/aws-dataset/aws dataset/aws dataset/output-onlinetsvtools.csv")
df_test.head()
test = pd.DataFrame()
test['review'] = df_test.benefitsReview
test.to_csv("test2.csv", header=False,index=False)
import json

from pprint import pprint



with open("../input/output/predictions.jsonl", "r") as f:

    f = f.readlines()
pprint(f)
predictlabels = []

for i in f:

    j= json.loads(i)["Classes"]

    predictlabels.append([j[0]["Name"]])
pprint(predictlabels)
test_input = pd.read_csv("../input/test-data/test2.csv")
test.head()
print(test_input.head())