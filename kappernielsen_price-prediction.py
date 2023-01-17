import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/train.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")
df_train = pd.DataFrame(data_train)
df_test = pd.DataFrame(data_test)
df_train.head(n=50)