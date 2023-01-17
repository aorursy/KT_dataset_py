!pip install jcopml
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer



from jcopml.pipeline import num_pipe, cat_pipe

from jcopml.utils import save_model, load_model

from jcopml.plot import plot_missing_value

from jcopml.feature_importance import mean_score_decrease

df = pd.read_csv("../input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv",index_col='Unnamed: 0')

df.head()
df.type.value_counts() #untuk melihat apa saja dari kolom tersebut
df.drop(columns=["type"],inplace=True)

df.head()
pd.options.display.max_rows = 1000 
df
plot_missing_value(df)
df= df[~df.IMDb.isna()]

df= df[~df.Age.isna()]

df= df[~df["Rotten Tomatoes"].isna()]


plot_missing_value(df)
plt.figure(figsize=(15,10))



plt.subplot(241)

sns.countplot("Netflix", data=df)



plt.subplot(242)

sns.countplot("Hulu", data=df)



plt.subplot(243)

sns.countplot("Prime Video", data=df)



plt.subplot(244)

sns.countplot("Disney+", data=df)
plt.figure(figsize=(15,10))



plt.subplot(221)

plt.hist(df.Age)

plt.title("Age")



plt.subplot(222)

plt.hist(df.IMDb)

plt.title("IMDb")
df.Age.value_counts()
df.IMDb.value_counts()
df["total"] = df.Netflix + df.Hulu + df["Prime Video"] + df["Disney+"]

df.head()
cc= df.sort_values("IMDb", ascending=False)

cc.head(10)