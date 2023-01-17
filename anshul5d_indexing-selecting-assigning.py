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
import pandas as pd
reviews['description']
# Your code here
reviews['description'].head(1)
# Your code here
reviews.loc(1)
# Your code here
reviews['description'].head(10)

# Your code here
reviews[reviews.index.isin([1,2,3,5,8])]
# Your code here
reviews[reviews[['country','province','region_1','region_2']].index.isin([0,1,100,1000])]
# Your code here
reviews[['country','variety']].iloc[0:1000]
# Your code here
reviews[reviews['country']=='Italy']
# Your code here
reviews[reviews['region_2'].notnull()]
# Your code here
reviews['points']
# Your code here
reviews['points'].iloc[0:1000]
# Your code here
reviews['points'].tail(1000)
# Your code here
reviews[reviews['country']=='Italy']['points']
# Your code here
rec=reviews[reviews['country'].isin(['France','Italy'])]
rec[rec['points']>=90]