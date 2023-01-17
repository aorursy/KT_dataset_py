import numpy as np

import pandas as pd



df = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")

for col in df.columns:

    df[col] = df[col].astype(str)

print(df.shape)

df.head()
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))



def evaluate(y_true, y_pred):

    score = 0

    for i in range(y_true.shape[0]):

        score += jaccard(y_true[i], y_pred[i])

    return score/y_true.shape[0]

    

evaluate(df["selected_text"].values, df["text"].values)
sub_df = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")

sub_df = sub_df.drop("sentiment", axis=1).rename(columns={"text": "selected_text"})

sub_df.head()
sub_df.to_csv("submission.csv", index=False)