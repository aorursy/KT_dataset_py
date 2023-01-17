import numpy as np 

import pandas as pd 
df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
df.head()
df.tail()
df.info()
df.describe()
type(df)
type(df.name)
df["price"].dtype
df["name"].dtype
df.name.dtype
df["name"] = df["name"].astype("category")

df.name.dtype
values = np.random.randn(50)

values
normal_distribution = np.random.normal(0,1, 50)

normal_distribution
data = pd.DataFrame({"A" : [1,2,3,4], 

                     "B" : [1.0, 2.2, 3.5, 5.4]

                    })

data
data.columns
data.A
data["A"]
data.iloc[ : , :1 ]
data.iloc[ : , 1 ]
data.loc[5] = [1, 6.1]

data
private = df["room_type"] == "Private room"

df[private].head()
df.query('room_type == "Private room"').head()
df[df["room_type"].isin(["Private room", "Entire home/apt"])].head()
rt = ["Private room", "Entire home/apt"]

df.query('room_type == @rt').head()
df.groupby("neighbourhood_group")["price"].mean()
df.groupby(["neighbourhood_group", "room_type"])["price"].mean()
df.groupby(["neighbourhood_group", "room_type"])["price"].agg(['mean', "sum"])
df.groupby(["neighbourhood_group", "room_type"])["price"].agg(['mean', "sum"]).reset_index()
df.columns
df.filter(["price", "host_name", "number_of_reviews"]).head()
df.filter(like = "neighbourhood").head()