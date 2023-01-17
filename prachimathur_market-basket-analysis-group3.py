# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



groceries=pd.read_csv("../input/market-basket-analysis/Groceries.csv",header=None)

groceries.head()
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt

import seaborn as sns

import wordcloud

from wordcloud import WordCloud,STOPWORDS



from wordcloud import WordCloud



plt.rcParams['figure.figsize'] = (10, 10)

wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121,stopwords=STOPWORDS.add("NaN")).generate(str(groceries[0]))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Items',fontsize = 20)

plt.show()
# looking at the frequency of most popular items 



plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.copper(np.linspace(0, 1, 40))

groceries[1].value_counts().head(40).plot.bar(color = color)

plt.title('frequency of most popular items', fontsize = 20)

plt.xticks(rotation = 90 )

plt.grid()

plt.show()
# making each customers shopping items an identical list

trans = []

for i in range(0, 7501):

    trans.append([str(groceries.values[i,j]) for j in range(0, 20)])



# conveting it into an numpy array

trans = np.array(trans)
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder



te = TransactionEncoder()

groceries = te.fit_transform(trans)

groceries = pd.DataFrame(groceries, columns = te.columns_)



# getting the shape of the data

groceries.shape
for col in groceries.columns: 

    print(col)
groceries.columns
groceries.drop(["nan", "`"], axis = 1, inplace = True)
for col in groceries.columns: 

    print(col)
groceries.head()
a=groceries*1

a.head()
b=groceries.sum(axis = 0, skipna = True)

b=pd.DataFrame(b)
b.columns = ['frequency']
b.head()
b.plot.bar()

plt.title('Frequency of all items purchased by customers')

plt.xlabel('Items')

plt.ylabel('count')

plt.figure(figsize=(200,20))

from pylab import rcParams

rcParams['figure.figsize'] = 10, 5



plt.show()
#FREQUENCY PLOT



c=b.sort_values('frequency',ascending=False).head(30)



c.plot.bar()

rcParams['figure.figsize'] = 10, 5



plt.title('Frequency of top 30 popular items')

plt.xlabel('Top 30 popular items')

plt.ylabel('count')

plt.show()
c['items'] = c.index

c.head()

c=pd.DataFrame(c)

c.head()
d=b.sort_values('frequency',ascending=False)

d.head()



d['items'] = d.index

d.head()

d=pd.DataFrame(d)
d.shape
#WORDCLOUD



import matplotlib.pyplot as plt

from wordcloud import WordCloud,ImageColorGenerator



wordcloud = WordCloud( background_color="white",width = 800,  height = 800)

wordcloud.generate_from_frequencies(frequencies=d.frequency)

plt.figure(figsize=(15,15))



plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.title('MOST POPULAR ITEMS',fontsize = 20)

plt.show()