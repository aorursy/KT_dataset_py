import numpy as np

import pandas as pd



pd.set_option('display.max_columns', None)  # or 1000

pd.set_option('display.max_rows', 100)  # or 1000

pd.set_option('display.max_colwidth', 100)  # or 199
train = pd.read_csv("../input/data-science-bowl-2019/train.csv")

train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")
display(train_labels[train_labels['game_session'] == '47e17c338e0ee335'])
temp = train[train['game_session'] == '47e17c338e0ee335']
display(temp)
temp = train[train['game_session'] =='901acc108f55a5a1']
display(temp)
temp = train[train['game_session'] =='a9c6f88dee27142c']
display(temp)
display(train_labels[train_labels['game_session'] == 'a9c6f88dee27142c'])