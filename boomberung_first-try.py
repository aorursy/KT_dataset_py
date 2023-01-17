import pandas as pd
import numpy as np
import matplotlib 
from sklearn.ensemble import GradientBoostingRegressor

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_len = train_df.shape[0]

both_df = train_df.append(test_df)
both_df = both_df.drop(labels = ["Id"],axis = 1)
## Separate train dataset and test dataset

train = both_df[:train_len]
test = both_df[train_len:]
test.drop(labels=["winPlacePerc"],axis = 1,inplace=True)

## Separate train features and label 

train["winPlacePerc"] = train["winPlacePerc"].astype(int)

y = train["winPlacePerc"]

X = train.drop(labels = ["winPlacePerc"],axis = 1)
clf = GradientBoostingRegressor()
clf.fit(X,y)
y_test = clf.predict(test)
temp = pd.DataFrame(pd.read_csv("../input/test.csv")['Id'])
temp['winPlacePerc'] = y_test
temp.to_csv("submission.csv", index = False)