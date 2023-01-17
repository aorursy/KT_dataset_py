import pandas as pd



train = pd.read_csv("../input/train.csv")

train
train.groupby("is_duplicate")["id"].count().plot(kind="bar")