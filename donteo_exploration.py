%matplotlib inline

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk


# Input data files are available in the "../input/" directory.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Reviews.csv", index_col = 0)
print("Number of reviews = {}".format(len(df)))
df.head()
df.ProductId.value_counts().head()
df[df.ProductId=='B007JFMH8M'][['Score','Summary','Text']].head()
df.ix[1]

df.Score.value_counts()
df['datetime'] = pd.to_datetime(df["Time"], unit='s')
df.groupby([df.datetime.dt.year, df.datetime.dt.month]).count()['ProductId'].plot(kind="bar",figsize=(30,10))
# Start with the summary data
df.Summary = df.Summary.astype(str)
downcase = lambda x: x.lower()
df.Summary.apply(downcase)
df.Summary = df.Summary.astype(str)
df.Summary.ix[3]
df.dtypes
