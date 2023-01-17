import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
check_q1(reviews["points"].median())
check_q2(reviews["country"].unique())
check_q3(reviews["country"].value_counts())
price_median = reviews["price"].median()
check_q4(reviews["price"].map(lambda x: x - price_median))
check_q4(reviews["price"].apply(lambda x: x - price_median))
check_q5(reviews.loc[(reviews["points"] / reviews["price"]).idxmax()]["title"])
tropical = reviews["description"].map(lambda x: "tropical" in x).value_counts()[True]
fruity = reviews["description"].map(lambda x: "fruity" in x).value_counts()[True]
check_q6(pd.Series([tropical, fruity], index=["tropical", "fruity"]))
reviews_not_null = reviews.loc[(reviews["country"].notnull()) & 
                               (reviews["variety"].notnull())]
country_varieties = df.apply(lambda x: x["country"] + 
                             " - " + 
                             x["variety"], axis=1)
check_q7(country_varieties.value_counts())
