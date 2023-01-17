#importing libraries

import pandas as pd 

import seaborn as sns
#reading dataset

df=pd.read_csv("../input/youtube_dataset.csv")
df.head()
#shuffle data

df.sample(frac = 1)
#sort values wrt number of likes

df.sort_values("Likes",ascending=False)