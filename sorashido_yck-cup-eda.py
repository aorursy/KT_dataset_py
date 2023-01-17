import numpy as np
import pandas as pd
import os

import seaborn as sns
sns.set(rc={'figure.figsize':(9,7)})
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
os.listdir("../input/ykc-cup-1st/")
train = pd.read_csv("../input/ykc-cup-1st/train.csv")
test = pd.read_csv("../input/ykc-cup-1st/test.csv")
train.head()
test.head()
train.describe()
train.fillna(np.nan).isnull().sum()
train["store_id"].nunique()
test["store_id"].nunique()
venn2([set(train["store_id"]), set(test["store_id"])], ('train', 'test'))
g = sns.barplot(x="day_of_week",y="log_visitors",data=train)
g = g.set_ylabel("log_visitors")
train["genre_name"].unique()
g = sns.barplot(x="genre_name",y="log_visitors",data=train)
g = g.set_ylabel("log_visitors")
train["area_name"].nunique()
train["prefecture"] = train["area_name"].str.split(expand=True)[0]
g = sns.barplot(x="prefecture",y="log_visitors",data=train)
g = g.set_ylabel("log_visitors")
g = sns.kdeplot(train["log_visitors"], color="Red", shade=True)
g.set_ylabel("Frequency")
train["weekend"] = ((train["day_of_week"] == "Sunday") | (train["day_of_week"] == "Saturday"))
g = sns.kdeplot(train["log_visitors"][train["weekend"] == True], color="Red", shade=True)
g = sns.kdeplot(train["log_visitors"][train["weekend"] == False], color="Blue", shade=True)
g.set_ylabel("Frequency")
