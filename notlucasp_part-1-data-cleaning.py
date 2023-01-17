import numpy as np

import pandas as pd
df1 = pd.read_csv("/kaggle/input/financial-news-headlines/cnbc_headlines.csv")

df1 = df1.dropna()

df1 = df1.drop_duplicates(subset=['Headlines', 'Description'], keep='first')

df1.reset_index(drop=True, inplace=True)

df1
df1.info()
def replace_dt(s):

    s = s.replace("Sept", "Sep").replace("March", "Mar").replace("April", "Apr").replace("June", "Jun").replace("July", "Jul")

    if s[0].isspace():

        s = s.replace(" ", "0", 1)

    s = s.replace(",  ", ", 0", 1)

    return s
from datetime import datetime

f = '%I:%M  %p ET %a, %d %b %Y'

dates = []

times = []

for item in df1.iloc[:, 1].values:

    item = replace_dt(item)

    dates.append(datetime.strptime(item, f).strftime("%m-%d-%Y"))

    times.append(datetime.strptime(item, f).strftime("%H:%M:%S"))
df1['Date'] = dates

df1["Date"] = df1["Date"].astype("datetime64")

df1['Time'] = times

df1 = df1[["Date", "Time", "Headlines", "Description"]]

df1
# Storing data for later use (EDA, NLP, ANN and RNN)

%store df1
df2 = pd.read_csv("/kaggle/input/financial-news-headlines/reuters_headlines.csv")

df2 = df2.dropna()

df2 = df2.drop_duplicates(subset=['Headlines', 'Description'], keep='first')

df2.reset_index(drop=True, inplace=True)

df2
df2["Time"] = df2["Time"].astype("datetime64")

df2 = df2[["Time", "Headlines", "Description"]]

df2.rename(columns={"Time":"Date"}, inplace = True)

df2
df2.info()
# Storing data

%store df2
df3 = pd.read_csv("/kaggle/input/financial-news-headlines/guardian_headlines.csv")

df3 = df3.dropna()

df3 = df3.drop_duplicates(subset=['Headlines'], keep='first')

df3.reset_index(drop=True, inplace=True)

df3
df3["Time"] = pd.to_datetime(df3["Time"], errors = 'coerce')

df3.rename(columns={"Time":"Date"}, inplace = True)

df3
df3.info()
# Storing data

%store df3