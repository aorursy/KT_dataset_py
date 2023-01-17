import pandas as pd
from IPython.display import YouTubeVideo

train = pd.read_json('../input/train.json')
train_suspect = train[train.index==201]
print(train_suspect.iloc[0, 1:])
YouTubeVideo(train_suspect.iloc[0, 4],start=train_suspect.iloc[0, 3],end=train_suspect.iloc[0, 1])