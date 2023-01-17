import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import os
train=pd.read_csv('../input/train.tsv', sep='\t', encoding='utf-8')

test=pd.read_csv('../input/test.tsv', sep='\t', encoding='utf-8')

sample = pd.read_csv('../input/sample_submission.csv', sep='\t', encoding='utf-8')
train.head()
test.head()
sample.head()
med = train["price"].median() # temporary price.

h = test.copy()

h["price"]=med

h = h[["test_id","price"]]

h.to_csv("output.csv", index =False)

h