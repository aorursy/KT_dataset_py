import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
train_df = pd.read_csv("../input/train.csv")
df = train_df.sample(frac=0.02)
df.info()
# get_difficulty_report takes two arguments texts and their labels.
sents = df["comment_text"].values
df['toxic_score'] = df.iloc[:,2:-1].sum(axis=1)
labels = df['toxic_score'].values
#print difficulty report
from edm import report
print(report.get_difficulty_report(sents, labels))