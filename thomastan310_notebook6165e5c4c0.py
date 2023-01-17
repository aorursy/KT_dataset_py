import numpy as np

import pandas as pd

train =pd.read_csv("../input/train.csv")

test  =pd.read_csv("../input/test.csv")
train.head()
test.head()
test["Survived"] = 0
submission = test.loc[:,["PassengersId", "Survived"]]
submission.head()