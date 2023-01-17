import os

import pandas as pd

import missingno as msno

#for dirname, _, filenames in os.walk("/kaggle/input"):

 #   for filename in filenames:

  #          print(os.path.join(dirname,filename))

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")



print(train.shape)

print(train.isnull().sum())

msno.bar(train)