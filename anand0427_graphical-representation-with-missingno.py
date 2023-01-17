import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno as msno
wiki_df = pd.read_csv("../input/wiki_movie_plots_deduped.csv")
wiki_df.head()
wiki_df.shape
wiki_df.describe()
wiki_df.info()
wiki_df["Director"] = wiki_df["Director"].apply(lambda x: np.NaN if x == "Unknown" else x)
wiki_df["Cast"] = wiki_df["Cast"].apply(lambda x: np.NaN if x == "NaN" else x)
wiki_df["Genre"] = wiki_df["Genre"].apply(lambda x: np.NaN if x == "unknown" else x)
wiki_df.info()
msno.matrix(wiki_df)
# wiki_df.Cast
wiki_df[wiki_df["Title"].duplicated(keep="first")==True].head()
wiki_df[wiki_df["Wiki Page"].duplicated(keep="first")==True].head()
msno.dendrogram(wiki_df)
msno.heatmap(wiki_df)
msno.bar(wiki_df)