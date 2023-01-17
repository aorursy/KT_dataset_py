import json, os, sys

import pandas as pd
host = "https://www.kaggle.com"
N_PAGES = 500
page_list = list(range(1, N_PAGES + 1))
def page_path(page):

    return f"topics.json?sortBy=hot&group=all&page={page}&pageSize=20&category=all&kind=all"
with open("urls.txt", "w") as f:

    for page in page_list:

        print(f"{host}/{page_path(page)}", file=f)
!wget -w 2 -i urls.txt -o wget.log
def load_page(page):

    fn = page_path(page)

    with open(fn) as f:

        j = json.load(f)

        s = [pd.Series(e) for e in j["topics"]]

        df = pd.concat(s, 1).T

        df.insert(0, "Page", page)

        return df
dfs = [load_page(p) for p in page_list]
df = pd.concat(dfs).reset_index(drop=True)

df.shape
users = df.pop("userAvatar").apply(pd.Series)
df = df.join(users.add_prefix("author_"))

df.shape
df.columns
df.authorType.value_counts()
date_tag = pd.datetime.now().strftime("%Y-%m-%d")

date_tag
df.to_csv(f"discussion-hotness-{date_tag}.csv", index_label="Index")
!zip -m -9 pages topics.* >>compress.log