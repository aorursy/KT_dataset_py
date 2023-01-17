!ls ../input
import pandas as pd

df = pd.read_csv("../input/train.csv")

df= df.loc[:50000]

df.to_csv("dev_toxic.csv", index=False)
test = pd.read_csv("../input/test.csv")

print("the shape is {}".format(test.shape))
test= test.loc[:25000]

test.to_csv("test_toxic.csv", index=False)