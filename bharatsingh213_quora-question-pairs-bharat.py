import pandas as  pd
import os

print(os.listdir("../input"))
df_train=pd.read_csv("../input/train.csv", nrows=3000)

df_test=pd.read_csv("../input/test.csv", nrows=3000)
df_train.head()
df_test.head()