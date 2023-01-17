# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#lets take a look at that categorical data--it has been giving me fits!

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import collections #I had a list of lists going, the good people at Stack Overflow said DefaultDict!

import bokeh

from bokeh.plotting import output_notebook

output_notebook(bokeh.resources.INLINE)

data=pd.read_csv("../input/train_categorical.csv",chunksize=100000, dtype=str,usecols=list(range(1,2141)))

uniques = collections.defaultdict(set)





for chunk in data: 

    for col in chunk:

        uniques[col] = uniques[col].union(chunk[col][chunk[col].notnull()].unique())

# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.
def numerifyCategories(x):

    """ Converts category defect strings to numbers"""

    if x:

        if isinstance(x, str):

            return len(x)

        else:

            return x

    else:

        return 0





def findResponse(x):

    """ Find the Response value from Id_Response mapping csv"""

    response_df = pd.read_csv('../input/Id_Response.csv', index_col='Id')

    if x in response_df.index:

        return float(response_df.loc[x]['Response'])

    else:

        return 0.0
def scatterplot(scatterDF, xcol, ycol, xlabel, ylabel, group=None, **kwargs):

    from bokeh.charts import Scatter

    if not group:

        scatter = Scatter(scatterDF, x=xcol, y=ycol, xlabel=xlabel, ylabel=ylabel, **kwargs)

    else:

        scatter = Scatter(scatterDF, x=xcol, y=ycol, xlabel=xlabel,

                                ylabel=ylabel, color=group, **kwargs)

    return scatter
#Are any variables empy?

emptyCols

empty=0

for key in uniques:

    if len(uniques[key])==0:

        print(key)

        empty=empty+1
#how bout columns with a single value?

single=0

for key in uniques:

    if len(uniques[key])==1:

        print(key,uniques[key])

        single=single+1
#how about multi-valued keys?

multi=0

for key in uniques:

    if len(uniques[key])>1:

        print(key,uniques[key])

        multi=multi+1
import matplotlib

import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt

 

objects = ('Empty', 'Single Value', 'Multi-Value')

y_pos = np.arange(len(objects))

performance = [empty,single,multi]

 

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Usage')

plt.title('Number of Features in Category Data')

 

plt.show()
data = data.fillna(0).applymap(numerifyCategories)

data['Response'] = list(map(findResponse, trainCat['Id']))
scatterplot(data,)