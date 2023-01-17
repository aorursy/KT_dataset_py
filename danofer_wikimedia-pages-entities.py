import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from unidecode import unidecode # fast strip accents

import string



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
pages = pd.read_csv("/kaggle/input/kensho-derived-wikimedia-data/page.csv",dtype = {"title":"str"})

print(pages.shape)

pages.describe()
pages
pages.sort_values("views")
pages["title"] = pages["title"].astype(str)
pages.head(10)["title"].apply(unidecode)
display(pages["title"].head())

pages["title"] = pages["title"].apply(unidecode)

display(pages["title"])
print(pages.shape[0])

pages.drop_duplicates(["title"]).shape[0]
pages.sort_values("views")["title"]
print(string.punctuation)
## take a subset of punctuations to remove , or keep all of them? (what about "new-york" ?)

puncts = "!#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
a = pages.sort_values("views")["title"].replace("[']", "",regex=True).replace("[,.]", " ",regex=True)

a
# pages["title"].apply(unidecode).to_csv("wikipedia_pages.csv.gz",index=False,compression="gzip")



a.to_csv("wikipedia_pages.csv.gz",index=False,compression="gzip")