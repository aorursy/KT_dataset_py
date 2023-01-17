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

df.shape[0] - df.drop_duplicates(subset="content").shape[0]
df.drop_duplicates(subset="content",inplace=True)
print("df without duplicated content:",df.shape[0])
# how many unique authors?
# df.author.value_counts().shape[0]
df.author.nunique()
df.author.value_counts().head() # Top authors have tens of thousands of tweets
# df.author.value_counts(normalize=True).head() # top author are 1-2 % of all tweets 
(df.author.value_counts()<5).sum() # a few hundred with only a few posts
df.language.value_counts()
df_en = df.loc[df.language=="English"]
print(df_en.shape[0])
df_en.describe(include="all")
df = df.loc[df.account_category != "Unknown" ]
df.account_category.value_counts(normalize=True)
df_en.account_category.value_counts()
print("original Unknown counts (for english only tweets)")
df_en = df_en.loc[df.account_category != "Unknown" ]
df_en.account_category.value_counts(normalize=True)
## Model building & text features can go here:

df.to_csv("russianTweet538Election.csv.gz",index=False,compression="gzip")
df_en.sample(frac=0.25).to_csv("russianTweet538Election_eng_sample.csv.gz",index=False,compression="gzip")