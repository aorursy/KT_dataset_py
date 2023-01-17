import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import random
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train_insincere=list(train.loc[train.target==1,"question_text"])
train_sincere=list(train.loc[train.target==0,"question_text"])
def get_sincere(n=1):
    index=random.sample(range(len(train_sincere)),n)
    for i in index:
        print(train_sincere[i])
def get_insincere(n=1):
    index=random.sample(range(len(train_insincere)),n)
    for i in index:
        print(train_insincere[i])
get_insincere(n=2)
get_sincere()
