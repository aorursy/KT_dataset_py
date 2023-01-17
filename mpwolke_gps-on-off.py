#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://images.unsplash.com/photo-1509786679719-3d066e68f607?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60',width=400,height=400)
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

Image(url = 'https://images.unsplash.com/photo-1515941821061-27d7f9900d43?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60',width=400,height=400)
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadsgpscsv/gps.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'gps.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
df.head()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://images.unsplash.com/photo-1545631903-b34f49c1ada9?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60',width=400,height=400)
df.describe()
df.info()
categorical_cols = [cname for cname in df.columns if

                    df[cname].nunique() < 10 and 

                    df[cname].dtype == "object"]





# Select numerical columns

numerical_cols = [cname for cname in df.columns if 

                df[cname].dtype in ['int64', 'float64']]
print(numerical_cols)
#Missing values. Codes from my friend Caesar Lupum @caesarlupum

total = df.isnull().sum().sort_values(ascending = False)

percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)

missing_data  = pd.concat([total, percent], axis=1, keys=['estacao', 'conex_gps'])

missing_data.head(8)
# Number of each type of column

df.dtypes.value_counts()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://images.unsplash.com/photo-1509576931792-214960705f8a?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60',width=400,height=400)
import matplotlib.pyplot as plt

import seaborn as sns
sns.scatterplot(x='estacao',y='geocodigo',data=df)
sns.countplot(df["estacao"])
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('estacao').size()/df['uf'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
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

Image(url = 'https://images.unsplash.com/photo-1510333337682-fdd0eba357a4?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60',width=400,height=400)
import folium
import geopandas as gpd



from learntools.core import binder

binder.bind(globals())

df.plot()
# How many rows in each column have missing values?

print(df.isnull().sum())



# View rows with missing locations

rows_with_missing = df[df["estacao"]=="conex_gps"]

rows_with_missing
import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



init_notebook_mode(connected=True) 



import pandas_profiling as pp

import plotly.express as px
import folium

from folium import Marker, GeoJson

from folium.plugins import HeatMap



import pandas as pd

import geopandas as gpd



# Function for displaying the map

def embed_map(m, file_name):

    from IPython.display import IFrame

    m.save(file_name)

    return IFrame(file_name, width='100%', height='500px')
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://images.unsplash.com/photo-1511068797325-6083f0f872b1?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://images.unsplash.com/photo-1516546453174-5e1098a4b4af?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://images.unsplash.com/flagged/photo-1554692938-b59814c27db8?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60',width=400,height=400)