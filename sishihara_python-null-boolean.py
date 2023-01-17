import pandas as pd





train = pd.read_csv('../input/titanic/train.csv')
train.head(10)
train.Age==False
train[train.Age==False]
train = train[train.Age==False]

train