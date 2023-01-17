import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
reviews['points'].median()
reviews['country'].unique()
reviews['country'].value_counts()
reviews.price.map(lambda p:p-reviews.price.median())
reviews.loc[(reviews.points / reviews.price).idxmax()].title
(reviews['province']==reviews['region_1']).value_counts()
s1=reviews.description.str.contains('tropical').value_counts()
s2=reviews.description.str.contains('fruity').value_counts()
ser=pd.Series([s1[True],s2[True]],index=['tropical','fruity'])
check_q7(ser)
cwn=reviews.dropna(subset=['country','variety'])
ser=cwn.country+" - "+cwn.variety
check_q8(ser.value_counts())
