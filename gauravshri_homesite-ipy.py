import pandas as pd
import statsmodels

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
print(test.shape)

print(train.head(n=5))
print(test.head(n=5))

train.columns
statsmodels.s