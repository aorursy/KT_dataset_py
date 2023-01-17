import pandas as pd
# from kaggle.competitions import nflrush
# # env = nflrush.make_env()
# from kaggle.
train = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv", low_memory=False)
import pandas_profiling
viz = pandas_profiling.ProfileReport(train)
viz.to_widgets()
import missingno as msno
msno.matrix(train)
msno.bar(train)
for col in range(len(train.columns)):
    n = train.iloc[:, col].isnull().sum()
    if n > 0:
        print(list(train.columns.values)[col] + ": "+ str(n) + " Nan")
pd.set_option('max_columns', 100)
train
train

