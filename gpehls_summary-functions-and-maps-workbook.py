import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews
check_q2(pd.DataFrame())
reviews.head()
check_q1(reviews.points.median())
check_q2(reviews.country.unique())
check_q3(reviews.country.value_counts())
median = reviews.price.median()
check_q4(reviews.price.map(lambda v: v - median))
median = reviews.price.median()
check_q5(reviews.price.apply(lambda v: v - median))
check_q5(reviews.loc[(reviews.points / reviews.price).idxmax()].title)
tropical = sum(reviews.description.map(lambda v:v.count('tropical')))
fruity = sum(reviews.description.map(lambda v:v.count('fruity')))
s_ = pd.Series(name='counting tropical and fruity',data=[tropical, fruity], index=['Tropical','Fruity'])
s_
df_=pd.DataFrame(reviews.loc[reviews.country.notnull() & reviews.variety.notnull()])
s_=pd.Series(name='counting variety members per country', data=df_.country +" - "+ df_.variety)
#sum(s_.map(lambda v:v.count(s_.unique())))
s_.value_counts()
#print(answer_q7())