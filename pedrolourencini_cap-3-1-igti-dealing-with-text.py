import pandas as pd

import numpy as np

import datetime
df_flickr = pd.read_csv('../input/bl-flickr-images-book/BL-Flickr-Images-Book.csv')
df_flickr.head()
df_flickr.shape
df_flickr.info()
#cleaning dataset

column_rem = ["Edition Statement", "Corporate Author", "Corporate Contributors", "Former owner", "Engraver", "Contributors", "Issuance type", "Shelfmarks"]

df_flickr.drop(columns = column_rem, axis = 1, inplace = True)
df_flickr.info()
df_flickr.head()
print(df_flickr["Identifier"].is_unique) #to verify if all sections are unique
df_flickr["Identifier"].nunique()
df_flickr.loc[1905:, "Date of Publication"].head(10)
expre_regular = r'^(\d{4}|$)' #first 4 digits

extr = df_flickr["Date of Publication"].str.extract(expre_regular, expand = False)
extr[1905:]
df_flickr["Date of Publication"] = pd.to_numeric(extr)
df_flickr["Date of Publication"].head(10)
df_flickr.info()
df_flickr["Date of Publication"].isnull().sum()
df_flickr.dropna(subset=["Date of Publication"], inplace = True)

df_flickr["Date of Publication"] = df_flickr["Date of Publication"].astype(int)
df_flickr["Place of Publication"].head(10)
df_flickr["Place of Publication"].str.contains("London").head(10)
london = df_flickr["Place of Publication"].str.contains("London")

oxford = df_flickr["Place of Publication"].str.contains("Oxford")
df_flickr["Place of Publication"] = np.where(london, "London", np.where(oxford, "Oxford", df_flickr["Place of Publication"].str.replace("-", " ")))
df_flickr.head(10)