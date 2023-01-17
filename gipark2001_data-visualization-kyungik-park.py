import pandas as pd

reviews = pd.read_csv("../input/winemag-data_first150k.csv", index_col=0)
reviews.head()
reviews.columns
reviews[reviews['price']<100].sample(100).plot.scatter(x='price', y='points')
reviews[reviews['price']<100].plot.scatter(x='price', y='points')
reviews[reviews['price']<100].plot.hexbin(x='price', y='points', gridsize=15)