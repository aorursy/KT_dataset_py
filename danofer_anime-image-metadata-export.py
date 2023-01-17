import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
def binarize_cols(df,cols=['rating']):
    for col in cols:
        df[col] = df[col].map({'s':0, 'q':1})
    return df
df = pd.read_csv("../input/all_data.csv",usecols=['created_at', 'score', 'sample_width',
       'sample_height', 'preview_url', 'tags']).drop_duplicates().dropna()
df.created_at = pd.to_datetime(df.created_at,unit="s",infer_datetime_format=True)
print(df.shape)
print(df.columns)
df.head()
df.score.describe()
df.score.quantile(0.999)
# df["cat_score"] = pd.cut(df.score, 3, labels=["good", "medium", "bad"])
df["score"] = df.score.clip(lower=-1, upper=2)
df["score"].value_counts(normalize=True)
# downsample majority class -  zero score
df = pd.concat([df.loc[df.score==0].sample(frac=0.5),df.loc[df.score!=0]])
print(df.shape)
df["score"].value_counts(normalize=True)
df.to_csv("safeBooru_animeTags.csv.gz",index=False,compression="gzip")