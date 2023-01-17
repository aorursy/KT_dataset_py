import pandas as pd
import graphlab

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print(train.describe())

