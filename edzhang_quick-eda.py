import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train = pd.read_csv("../input/train.csv")
resource = pd.read_csv('../input/resources.csv')
test = pd.read_csv("../input/test.csv")
print("# of example in train set %d" % len(train) )
print("# of example in test set %d" % len(test) )
train.dtypes
train.sample(2)
resource.sample(5)
