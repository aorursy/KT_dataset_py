import numpy as np

import pandas as pd

import datetime



%matplotlib inline

pd.set_option('display.max_rows', 500)

df = pd.read_csv("../input/ratings_small.csv")



def timestamp_th(timestamp):

    dt = datetime.datetime.fromtimestamp(timestamp)

    return dt.year

df['Aeg'] = df['timestamp'].apply(timestamp_th)



df2 = df[["rating","Aeg"]]

df2.sort_values("Aeg", ascending=True)



a = round(df2.groupby("Aeg").mean(),2)

b = df2.groupby("Aeg").count()

a.columns = ["Keskmine hinne"]

b.columns = ["Hinnete arv"]

print(pd.concat([a, b], axis = 1))