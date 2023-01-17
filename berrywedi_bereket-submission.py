import pandas as pd

import os
os.getcwd()
df=pd.read_csv("../input/train.csv")
df.head(5)
df.info()
df.tail()
# we can manupilate the data

df['Age']
df=pd.read_csv("../input/train.csv")
import pandas as pd

import os
df=pd.read_csv("../input/train.csv")
df.Age[0:10]