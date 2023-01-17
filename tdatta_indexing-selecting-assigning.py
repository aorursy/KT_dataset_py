import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
import pandas as pd

reviews.description


import pandas as pd

reviews['description'] [0]

import pandas as pd

reviews.iloc[0]

import pandas as pd

reviews.iloc[:10,1]
import pandas as pd

reviews.iloc[[1,2,3,5,8]]
import pandas as pd

reviews.loc[[0,1,10,100], ['country', 'province', 'region_1', 'region_2']]
import pandas as pd

reviews.loc[:99, ['country', 'variety']]
reviews.loc[reviews.country == 'Italy']
reviews.loc[reviews.region_2.notnull()]
reviews.loc[:, 'points']

reviews.loc[:999, 'points']
reviews.iloc[-1000:]['points']
reviews.points[reviews.country == 'Italy']
reviews.loc[(reviews.country.isin(['Italy', 'France'])) & (reviews.points >= 90)]