import pandas as pd
pd.set_option('max_rows', 10)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
reviews.points.median()

reviews.country.unique()
reviews.country.value_counts().head(1)
medianPrice=reviews.price.median()
reviews.price.fillna(0).map(lambda x: x-medianPrice)
bestWineIndex=reviews.apply(lambda x:x.points/x.price,axis=1).idxmax()
reviews.iloc[bestWineIndex,:]
tcnt=reviews.description.map(lambda x: x.find('tropical')>=0).value_counts()[1]
fcnt=reviews.description.map(lambda x: x.find('fruity')>=0).value_counts()[1]
#tcnt+fcnt
fs=pd.Series([tcnt,fcnt],index=['tropicalCount','fruityCount'])
fs.sum()

df=reviews[['country','variety']].dropna()
df.apply(lambda x:x[0]+'-'+x[1],axis=1).value_counts()