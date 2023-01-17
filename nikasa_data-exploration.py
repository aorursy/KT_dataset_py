import pandas as pd
import numpy as np
import seaborn as sns
from math import log
from matplotlib import pyplot as plt
import re
df = pd.read_csv("../input/kaggledays-warsaw/train.csv", sep="\t", index_col='id')
# Convert dtype to correct format
coltypes = {"question_id":"str", "subreddit":"str", "question_utc":"datetime64[s]",
            "question_text": "str", "question_score":"int64", "answer_utc":"datetime64[s]",
            "answer_text":"str", "answer_score":"int64"}
for col, coltype in coltypes.items():
    df[col] = df[col].astype(coltype)

# Log transform all scores
logscales = ["question_score", "answer_score"]
for col in logscales:
    df[col] = df[col].apply(log)
# How Subreddit statistics affect answer_score?
sns.boxplot(x=df["subreddit"], y=df["answer_score"])
### How time Effect on the answer_score ?
df["answer_delay"] = (df.answer_utc - df.question_utc).dt.total_seconds().apply(lambda x: log(x+1))
df["answer_dow"] = df["answer_utc"].dt.dayofweek
df["answer_hod"] = df["answer_utc"].dt.hour
df["answer_tod"] = df["answer_utc"].dt.hour*24  + df["answer_utc"].dt.minute
df["time_trend"] = (df["answer_utc"] - df["answer_utc"].min()).dt.total_seconds()/3600

_, axes = plt.subplots(1, 3, figsize=(20, 8))
sns.regplot(x=df["answer_delay"], y=df["answer_score"], ax = axes[0], fit_reg=False)
sns.boxplot(x=df["answer_dow"], y=df["answer_score"], ax = axes[1])
sns.boxplot(x=df["answer_hod"], y=df["answer_score"], ax = axes[2])

_, ax = plt.subplots(1, 1, figsize=(20, 8))
# sns.jointplot(x=df["answer_tod"], y=df["answer_score"], kind="hex") 
sns.regplot(x=df["time_trend"], y=df["answer_score"], fit_reg=False, ax=ax)
# How question statistics affect answer_score
df_q = df.groupby("question_id").mean()
df_q["answer_count"] = df["question_id"].value_counts()
_, axes = plt.subplots(1, 3, figsize=(20, 8))
sns.regplot(x=df_q["question_score"], y=df_q["answer_score"], fit_reg=False, ax = axes[0])
sns.regplot(x=df_q["answer_count"], y=df_q["answer_score"], fit_reg=False, ax = axes[1])
sns.regplot(x=df_q["answer_delay"], y=df_q["answer_score"], fit_reg=False, ax = axes[2])
# How question time affect answer_score (Almost nothing)
df["question_hod"] = df["question_utc"].dt.hour
sns.boxplot(x=df["question_hod"], y=df["answer_score"])
# How content of the answer affects answer_score
url_regex = 'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
img_regex = 'https?:[^)''"]+\.(?:jpg|jpeg|gif|png)'

df["text_length"] = df["answer_text"].apply(lambda x: len(x))
df["answer_imgs"] = df["answer_text"].apply(lambda x: len(re.findall(img_regex, x))) #number of imgs in answer
df["answer_links"] = df["answer_text"].apply(lambda x: len(re.findall(url_regex, x))) #number of links  that are not imgs
df["answer_links"] = df["answer_links"] - df["answer_imgs"]
df.answer_imgs = df.answer_imgs.apply(lambda x: 6 if x > 6 else x)
df.answer_links = df.answer_links.apply(lambda x: 10 if x > 10 else x)

_, axes = plt.subplots(1, 3, figsize=(20, 8))
sns.regplot(x=df["text_length"], y=df["answer_score"], fit_reg=False, ax=axes[0])
sns.boxplot(x=df["answer_imgs"], y=df["answer_score"], ax=axes[1])
sns.boxplot(x=df["answer_links"], y=df["answer_score"], ax=axes[2])
