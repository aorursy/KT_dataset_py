import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.grouping_and_sorting import *
from IPython.display import display

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())

df = pd.DataFrame({'A': [1, 1, 2, 2],'B': [1, 2, 3, 4],'C': np.random.randn(4)})
display(df)
print(df.groupby('A'))
# Your code here
# common_wine_reviewers = _______
# check_q1(common_wine_reviewers)


import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.grouping_and_sorting import *
from IPython.display import display
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

x= reviews.groupby(['taster_twitter_handle']).taster_twitter_handle.count()
display(x)
#ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
#ans = ans.apply(lambda x: x.country + " - " + x.variety, axis='columns')
#ans.value_counts()
#display(reviews.groupby('taster_twitter_handle').taster_twitter_handle.count())
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.grouping_and_sorting import *
from IPython.display import display
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
x = reviews.groupby('price').points.max().sort_index()

display(x)
print(reviews.groupby('price').points.max().sort_index())
#print(answer_q2())
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.grouping_and_sorting import *
from IPython.display import display
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

x = reviews.groupby('variety').price.max()
y = reviews.groupby('variety').price.min()
print(x)
print('----------------------------------')
print(y)
x = pd.concat([x, y], axis = 1)
x.columns = ['Max Price', 'Min Price']
print('----------------------------------')
print(x)

reviews.groupby('variety').price.agg([min, max])
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.grouping_and_sorting import *
from IPython.display import display
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

x = reviews.groupby('taster_name').points.mean()
display(x)

print(answer_q4())
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.grouping_and_sorting import *
from IPython.display import display

x = reviews.groupby('variety').price.agg([max, min])

display(x.sort_values(by = 'max',ascending = False))
print('----------------------------------------------------')
display(x.sort_values(by = 'min',ascending = False))
#print(answer_q5())
#print(reviews.groupby('variety').price.agg([min, max]).sort_values(by=['min', 'max'], ascending=False))
# Your code here
# country_variety_pairs = _____
# check_q6(country_variety_pairs)