import pandas as pd 

import glob

import os

print(os.listdir("../input"))



PATH = "../input"
filenames = glob.glob(os.path.join(PATH, "*.csv"))

print(filenames)

df = pd.concat((pd.read_csv(f) for f in filenames))

print(df.shape)

df.head()
## I'll keep the external id for now. We drop the harvested ID. 

df.drop("harvested_date",axis=1,inplace=True)
# print(df.external_author_id.nunique()) # 2489

# df.external_author_id = pd.Categorical(df.external_author_id).codes

# print(df.external_author_id.nunique())  # # 2490



# # There's a mismatch - unknown where my bug is. Commenting out for now!
df.shape
df.dtypes
df.describe(include="all")
df=df.dropna()

df[df['content'].str.contains("ukushima")]['content'].values

df[df['content'].str.contains("hernobyl")]['content'].values
df[df['content'].str.contains("radioactiv")]['content'].values
df[df['content'].str.contains("CO2 ")]['content'].values

df[df['content'].str.contains("greenpeace")]['content'].values
df[df['content'].str.contains("climate")]['content'].values

df.shape[0] - df.drop_duplicates(subset="content").shape[0]
df.drop_duplicates(subset="content",inplace=True)

print("df without duplicated content:",df.shape[0])