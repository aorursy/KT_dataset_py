import pandas as pd

import numpy as np

from ggplot import *

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df_wine = pd.read_csv("../input/winemag-data-130k-v2.csv")
df_wine = df_wine.dropna(subset = ['price','country','variety'])
df_wine.columns
df_wine.info()
df_wine.isnull().sum()
df = df_wine[df_wine.variety.isin(df_wine.variety.value_counts().head(9).index)]

df = df[df.country.isin(df.country.value_counts().head(9).index)]

df.head()
p = ggplot(df,aes(x="points", y="price", shape ="variety", size ="price", color="country")) + geom_point()

p + facet_wrap('variety', scales="free_y") + xlab("points") + ylab("price") + ggtitle("winery review: country to price")
plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("winery review")

corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
from sklearn.model_selection import train_test_split



X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.3,

                                                    random_state=0,

                                                    stratify=y)
