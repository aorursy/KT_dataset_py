import pandas as pd

dataset = pd.read_csv('/kaggle/input/titanic/train.csv')
dataset["Age"][dataset["Age"].isnull()].index
dataset["Age"][dataset["Age"].isnull()]
[dataset["Age"].isnull()]
# 別の書き方

dataset.query('Age != Age').index