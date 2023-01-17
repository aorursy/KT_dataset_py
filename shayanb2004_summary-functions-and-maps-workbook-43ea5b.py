import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews.points.median()
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
print(reviews.country)
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews.country.value_counts()
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
medi = reviews.price.mean()
x = reviews.price.map(reviews.price - medi)
sorted(x.head())
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *
from IPython.display import display
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
y = reviews.points/reviews.price
y = y.to_frame()
data = pd.concat([y, reviews.title, reviews.country], axis = 1)
data.columns = ['Ratio', 'title', 'country']
#print(data.number.value_counts())
print(data.iloc[data['Ratio'].idxmax(skipna = True)]) 
x = max(data.Ratio.dropna())
y = data[data.Ratio==x]
#y.head()
display(y)
#print(data.title)
#print(x)
#print(data.map(lambda r: r==x))
#list_x = pd.concat([data, reviews.title], axis = 1)
#print(list_x)
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
list_tro = reviews.description.str.count('tropical')
list_trop = list_tro.sum()
list_fruity = reviews.description.str.count('fruity')
list_fruit = list_fruity.sum()
print('number of occurence of tropical keyword is :'+ str(list_trop))
print('number of occurence of fruity keyword is :'+ str(list_fruit))
print('-----------------------------------------------------')
#print(answer_q6())
tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
x = pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])
print(x)
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *
from IPython.display import display
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = ans.apply(lambda x: x.country + " - " + x.variety, axis='columns')
ans.value_counts()