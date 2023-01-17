#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSEAu9SAgQfkvMf0V7jiLI7vrCxbN2rg9pTLlQcJI-At2w49RS8',width=400,height=400)
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
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ6-X-GPtnQPTkn7H103z_N9v--5Ki4PgB4sVJDka9EE7ulH3x7',width=400,height=400)
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadsindigenacsv/indigena.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'indigena.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRZhpwaSyAkSBJ4jwW7Mwxz4SIIzzWzRUrbT1QoVaRTqJXR2HwK',width=400,height=400)
df.head()
df.info()
categorical_cols = [cname for cname in df.columns if

                    df[cname].nunique() < 10 and 

                    df[cname].dtype == "object"]





# Select numerical columns

numerical_cols = [cname for cname in df.columns if 

                df[cname].dtype in ['int64', 'float64']]
print(numerical_cols)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSuIFybdgj_wF2t_CxUsjpWf3KKlK5ZIpPO9QZJKhXi87SmCZ_b',width=400,height=400)
#Missing values. Codes from my friend Caesar Lupum @caesarlupum

total = df.isnull().sum().sort_values(ascending = False)

percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)

missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(8)
# Number of each type of column

df.dtypes.value_counts()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRIv0QvtxoCl9aVvn9yUsb17CKJfU5b9FL75hc3ZlV86P3KngRO',width=400,height=400)
import matplotlib.pyplot as plt

import seaborn as sns
sns.scatterplot(x='uf',y='pop_tot',data=df)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSOVyVv_k5c1Zt7WwrwH8puPa4eNjPaW_2Rm0DKmXe3ViDXrkFY',width=400,height=400)
sns.countplot(df["pop_tot"])
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('uf').size()/df['pop_tot'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT-8xVXxQ0gkj_7ZiNEdSBELDHEg0o-yBghicPe4iB3PirByWef',width=400,height=400)
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.uf)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="white").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRqR9R8snm5eX-WRgIp-crzYENJ5Q7lR6gOiCeDq1nuquhxTBN5',width=400,height=400)