import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(sorted(os.listdir("../input")))

pd.set_option('display.max_colwidth', -1)
%%time

useful_cols = ["username", "fullname", "tweet_url", "timestamp", "text"]

for f in sorted(os.listdir("../input")):

    if f == "Dr_Heba_Raouf.csv":

        df = pd.read_csv("../input/" + f)

    else:

        df = pd.read_csv("../input/" + f, sep=";", usecols=useful_cols)

        df["tweet_url"] = "https://twitter.com" + df["tweet_url"]

    split = df.text.str.split()

    hashtags = {}

    for row in split:

        if type(row) != list:

            continue

        for word in row:

            if word[0] == "#":

                if word not in hashtags:

                    hashtags[word] = 1

                else:

                    hashtags[word] += 1

    hashtags_by_popularity = sorted(hashtags.items(), key=lambda kv: kv[1], reverse=True)

    print("{} - {} tweets".format(f, len(df)))

    result_df = pd.DataFrame(hashtags_by_popularity, columns=["Hashtag", "Count"])

    display(result_df[:20])

    

    femin = df[df.text.str.contains(r"(?i)femin|(?i)haras ", na=False)]

    display(femin)