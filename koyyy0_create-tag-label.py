import json

import pandas as pd

decoder = json.JSONDecoder()

p=0

for i in range(17):

    json_log = []

    with open("../input/tagged-anime-illustrations/danbooru-metadata/danbooru-metadata/2017"+str(i).zfill(2)+".json", "r") as f:

        line = f.readline()

        while line:

            json_log.append(decoder.raw_decode(line))

            line = f.readline()

    c=[]

    print(len(json_log))

    for i in range(len(json_log)):

        if p==0:

            c.extend(json_log[i][0]["tags"])

        else:

            c.extend(json_log[i][0]["tags"])

    df=pd.DataFrame(c)

    if p==0:

        p=1

        df=df[~df.duplicated()]

        df1=df

    else:

        df1=pd.concat([df,df1])

        df1=df1[~df1.duplicated()]

df1["id"]=df1["id"].astype(int)

df1=df1.sort_values("id")

df1.reset_index(drop=True,inplace=True)
df1
"""Number of tag_labels"""

len(df1)
df1.to_csv("tag_labels.csv")