# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/blogtext.csv")
print(df.shape)
df.head()
df.drop_duplicates(subset="text",inplace=True)
df.text.str.len().describe()
df.text.str.len().plot()
df = df.loc[(df.text.str.len() < 18000) & (df.text.str.len() > 7)]
df.text.str.len().plot()
df.loc[df.text.str.len() < 30].shape
df.describe()
df.age.value_counts()
df.topic.value_counts()
df["word_count"] = df.text.str.split().str.len()
df["char_length"] = df.text.str.len()
df["id_count"] = df.groupby("id")["id"].transform("count")
df.head(12)
df.date = pd.to_datetime(df.date,errors="coerce",infer_datetime_format=True)
df.tail()
df.drop_duplicates(subset="id")["id_count"].describe()
df.shape
df = df.loc[df.id_count < 200]
df.shape
df.head()
df["age_group"] = pd.qcut(df["age"],3,precision=0,)
df.head()
df.age_group.value_counts()
df["age+sex"] = df["age_group"].astype(str)+df["gender"]
df.head()
df.info()
df["age+sex"].value_counts()
# df.to_csv("blogAuthors_sample.csv.gz",index=False,compression="gzip")
df.head()
df.drop(["age_group"],axis=1).drop_duplicates(subset="id",keep="first").to_csv("blogAuthors_sample_distinct.csv.gz",index=False,compression="gzip")
df[["id","date","topic","text","word_count"]].to_csv("blogAuthors_sample_history.csv.gz",index=False,compression="gzip")
