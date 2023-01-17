import pandas as pd

ramen = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv")

wine = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")
import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

plt.style.use('fivethirtyeight')



sns.countplot(np.round(ramen['Stars'].replace('Unrated', np.nan).dropna().astype(np.float64)))

plt.suptitle("Ramen Ratings")
plt.suptitle("Wine Ratings")

sns.countplot(np.round((wine['points'].dropna() - 80) / 4))
sns.countplot(

    np.round(ramen['Stars'].replace('Unrated', np.nan).dropna().astype(np.float64) * 2) / 2

)

plt.suptitle("Ramen Ratings")
sns.countplot(np.round((wine['points'].dropna() - 80) / 2) / 2)