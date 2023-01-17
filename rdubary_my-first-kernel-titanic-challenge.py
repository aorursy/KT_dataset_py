import pandas as pd

train =  pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.info()
train.shape
train.describe()
test.info()
test.shape
test.describe()
train['Survived'].value_counts(normalize = True)
train['Survived'][train['Sex'] == 'male'].value_counts(normalize = True)
train['Survived'][train['Sex'] == 'female'].value_counts(normalize = True)