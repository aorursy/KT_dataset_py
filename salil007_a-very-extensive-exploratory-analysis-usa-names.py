import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

from sklearn import preprocessing
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")
import os
# Any results you write to the current directory are saved as output.
# create a helper object for this dataset
usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")

# query and export data 
query = """SELECT year, gender, name, sum(number) as number FROM `bigquery-public-data.usa_names.usa_1910_current` GROUP BY year, gender, name"""
agg_names = usa_names.query_to_pandas_safe(query)
agg_names.to_csv("usa_names.csv")
agg_names.head()
agg_names.shape
pd.options.display.max_rows = 4000
agg_names.groupby('gender')['gender'].count().plot.bar()
temp = agg_names['gender'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud( max_font_size=50, 
                       stopwords=STOPWORDS,
                       background_color='black',
                       width=600, height=300
                     ).generate(" ".join(agg_names['name'].sample(2000).tolist()))

plt.figure(figsize=(14,7))
plt.title("Wordcloud for Top Keywords in Names", fontsize=35)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
agg_names.groupby('number')['number'].count().head().plot.bar()
temp = agg_names['number'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
temp = agg_names.groupby('year')['year'].count()
ye = temp
temp
q1=temp.plot.line(x=temp.index,y=temp.values,figsize=(12,3),lw=7)
q1.set_ylabel("Number of Applicants")
agg_names['name'].value_counts().head()
temp = agg_names.groupby('gender')['name'].value_counts()['M'].head()
temp
temp = agg_names.groupby('gender')['name'].value_counts()['F'].head()
temp
temp = agg_names.groupby('year')['gender'].value_counts()
temp = temp.loc[1::2]
mval = ye.values - temp.values
mval
q2=temp.plot.line(x=ye.index,y=mval,figsize=(12,3),lw=7,color="purple")
q2.set_ylabel("Number of Male Applicants")
temp = agg_names.groupby('year')['gender'].value_counts()
temp = temp.loc[1::2]
temp.values
q2=temp.plot.line(x=ye.index,y=temp.values,figsize=(12,3),lw=7,color='orange')
q2.set_ylabel("Number of Female Applicants")
temp = agg_names.groupby('year')['gender'].value_counts()
temp