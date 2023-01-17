# Installing Side Table

!pip install sidetable
import numpy as np

import pandas as pd

import sidetable

import ast

from sklearn.preprocessing import MultiLabelBinarizer
path = "../input/flipkart-products/flipkart_com-ecommerce_sample.csv"
df = pd.read_csv(f"{path}")

df.head()
df["product_category_tree"].apply(lambda x: ast.literal_eval(x)).apply(lambda x: len(x[0].split(">>"))).argmax()
df["product_category_tree"].iloc[1482]
category_levels = (df["product_category_tree"]

     .apply(lambda x: ast.literal_eval(x))

     .apply(lambda x: x[0].replace("\\","").replace("'","").split(" >> "))

     .apply(pd.Series)

     .rename(columns= lambda x: "level_"+str(x)))
df = pd.concat([category_levels, df], axis=1)
df.head()
print(f"Total Number of top level categories {df['level_0'].nunique()}")
df["level_0"].value_counts()[:10]
df["level_0"].value_counts(normalize=True)[:10]
df.stb.freq(["level_0"])[:10]
df.stb.freq(["level_0"],thresh=0.6, other_label="other_categories")
df.stb.freq(["level_0"], thresh=0.6) # without the other label
df[df["overall_rating"] != "No rating available"].stb.freq(["level_0"], thresh=0.6, other_label="other_rated_categories")
df[df["overall_rating"] != "No rating available"].stb.freq(["level_0"], value = "discounted_price", thresh=0.6)
(df[df["level_0"].isin(["Computers", "Clothing", "Jewellery"])]

    .groupby(["level_0", "level_1"]).agg({"discounted_price": ["sum"]}).stb.subtotal())
df.stb.missing()
df.stb.freq(["brand"], thresh=0.10, style=True)