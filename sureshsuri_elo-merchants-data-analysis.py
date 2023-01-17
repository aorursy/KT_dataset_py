import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv",parse_dates=["first_active_month"])
test=pd.read_csv("../input/test.csv",parse_dates=["first_active_month"])
print("columns and rows in train dataset:",train.shape)
print("columns and rows in test dataset:",test.shape)

train.head()
train.target.plot.hist()
hist=pd.read_csv("../input/historical_transactions.csv")
hist.head()
