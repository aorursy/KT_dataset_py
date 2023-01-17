import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
reviews['description']
# Your code here
reviews['description'].iloc[0]
# Your code here
reviews.iloc[0,:]
# Your code here
reviews['description'].iloc[0:10]
# Your code here
reviews.iloc[[1,2,3,5,8],:]
# Your code here
reviews.loc[[0,1,10,100],['country','province','region_1','region_2']]
# Your code here
reviews.loc[0:100,['country','variety']]
# Your code here
reviews[reviews['country']=='Italy']
# Your code here
reviews[reviews.region_2.isnull()]
# Your code here
reviews['points']
# Your code here
reviews['points'].iloc[0:1000]
# Your code here
reviews['points'].iloc[-1000:3]

#print(answer_q12())
# Your code here

reviews[reviews['country']=='Italy'].points
# Your code here
import numpy as np
#between_c=np.logical_and[['Italy'],['France']]
#review_point_range=reviews[between_c]
#review_point_range

reviews[reviews.points >= 90 & reviews['country'].isin(["Italy","France"])]

#print(answer_q14())