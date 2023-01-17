

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
df = pd.read_csv("../input/Library_Collection_Inventory.csv") #,nrows=5

print(df.shape)

df.head()
df2 = df.loc[df.ItemCount>1]

print(df2.shape[0])

df2.tail()
print(df2.ItemCount.describe())

df2.ItemCount.hist();
df2.loc[df.ItemCount>2].shape[0]
df2[df2.ItemCount>130].head()
# Except for the movie arrival, these popular items looks very weird. Likely indicates something funky with the data or counting? 
## acbk  = common code for book (there are others )
df2.loc[df2.ItemType=="acbk"].shape[0]

df2.loc[df2.ItemType=="acbk"].head(2)
# New DF for "popular books"

df_books = df2.loc[(df2.ItemType=="acbk") & (df2.ItemCount>3)]

print(df_books.shape[0])

print(df_books.ItemCount.describe())
df_books = df_books.sort_values("ItemCount",ascending=False).drop(["ReportDate"],axis=1).drop_duplicates("Title")

print(df_books.shape[0])

df_books.head()