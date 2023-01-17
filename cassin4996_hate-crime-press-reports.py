# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current/ directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
from IPython.core.display import HTML
import seaborn as sns
from plotly import __version__
import plotly.graph_objects as go
import cufflinks as cf
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
import plotly.express as px
init_notebook_mode(connected = True)
cf.go_offline()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS 

%matplotlib inline
a = pd.read_csv('/kaggle/input/classification-of-hate-crime-in-the-us/20170816_Documenting_Hate - Data.csv')
a.isnull().sum()
a.rename(columns = {'Article Title': 'ArticleDate','Organization': 'ArticleTitle','City': 'Organization', 'State' : 'City', 'URL' : 'State', 'Keywords': 'URL'}, inplace = True)
a.drop(['Article Date','Summary','Unnamed: 8'], axis = 1, inplace = True)
a['ArticleDate'] = pd.to_datetime(a['ArticleDate'])
a.ArticleDate = [d.date() for d in a.ArticleDate]
a.dtypes
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(a.loc[(a['City'].isnull() & a['State'].notna()) | (a['City'].notna() & a['State'].isnull()),['City','State']])
a.loc[[81,229,344,784,1622,1667,1737,1795,1848,2010,2498,2728,3307,3308,3342,3343],['State']] = 'DC'
a.loc[[379,509,555,611,767,775,781,854,937,1036,1051,1144,1360,1407,1551,1578,1697,1801,1882,1939,1959,2015,2165,2195,2197,2205,2316,2337,2344,2600,2636,2637,2712,2718,2960,3003,3011,3096,3170,3207,3290,3327,3428,3479,3573,3653],['State']] = 'NY'
a.loc[[150,298,718,1757,2573,3020],['City']] = 'New York'
a.loc[[410,1145,2725,2933,3164],['City']] = 'Baltimore'
a.loc[[558,3230,3483,3529],['City']] = 'Minneapolis'
a.loc[[822,2386,3599],['City']] = 'Los Angeles'
a.loc[[194,512,615],['City']] = 'Richmond'
a.loc[[1088,1998],['City']] = 'Lansing'
a.loc[[316],['State']] = 'New Mexico'
a.loc[[496],['City']] = 'Phoenix'
a.loc[[753],['State']] = 'NV'
a.loc[[834],['State']] = 'WI'
a.loc[[1057],['State']] = 'WA'
a.loc[[1298],['City']] = 'Salt Lake City'
a.loc[[1320],['State']] = 'New Mexico'
a.loc[[2125],['State']] = 'NJ'
a.loc[[2598],['State']] = 'New Mexico'
a.loc[[2850],['City']] = 'Houston'
states = { 'Connecticut' : 'CT', 'New York' : 'NY', 'Washington' : 'WA', 'Virginia' : 'VA','California' : 'CA', 'South Carolina' : 'SC', 'Colorado' : 'CO',
         'Indiana' : 'IN', 'Tennessee' : 'TN', 'Wisconsin' : 'WI', 'Illinois' : 'IL', 'Michigan' : 'MI' , 'Arizona' : 'AZ', 'Montana' : 'MT',
         'New Jersey' : 'NJ', 'Florida' : 'FL', 'Alaska' : 'AK', 'Texas' : 'TX', 'New Mexico' : 'NM', 'Utah' : 'UT', 'D.C.' : 'DC', 'Missouri' : 'MO',
         'Kentucky' : 'KY', 'District of Columbia' : 'DC', 'North Dakota' : 'ND', 'Louisiana' : 'LA', 'Ca' : 'CA','North Carolina' : 'NC', 'tx' : 'TX',
         'Maine' : 'ME', 'Iowa' : 'IA', 'ma' : 'MA', 'Venice Beach, CA 90291' : 'CA', 'Alabama' : 'AL', 'Nebraska' : 'NE', 'Maryland' : 'MD', 'ny' : 'NY',
         'Ohio' : 'OH', 'New York and California' : 'NY & CA', 'mi' : 'MI', 'Los Angeles, California' : 'CA', 'South Dakota' : 'SD', 'NORTH CAROLINA' : 'NC', 'Alexandria VA' : 'VA',
         'Oregon' : 'OR', 'Georgia' : 'GA', 'Kansas' : 'KS'}

b = []
for i,j in zip(states.keys(),states.values()):
    for k in a.index:
        if a.loc[k,'State'] == i:
            b.append(k)
    for k in b:
        a.loc[k,['State']] = j
    b.clear()
a.head()
fig = plt.figure(figsize = (20,10))
a.groupby('ArticleDate').agg('count')['ArticleTitle'].plot(kind = 'line')
plt.xlabel('Article Date', fontsize = 16)
plt.ylabel('Count of the articles published', fontsize = 16)
plt.suptitle('Line Graph - Article Date vs Number of Articles published in the particular period', fontsize = 20)
x = a.Organization.value_counts().head(30)
fig = px.bar(x = x.index, y = x)
fig.update_layout(
    title="Count of the Articles published by an Organization",
    title_x=0.5,
    xaxis_title="Organization",
    yaxis_title="Count of the Articles",
    )
fig.show()
fig = px.choropleth(locations=a.State.value_counts().index, locationmode="USA-states", color=a.State.value_counts().values,scope="usa")
fig.update_layout(
    title_text = "Number of Press-Reports made by a particular state's Media Organization",
    title_x = 0.5,
    geo_scope='usa',
)

fig.show()
c = []
for i in a.ArticleTitle:
    text_token = word_tokenize(i)
    text_token_without_sw = [word for word in text_token if not word in stopwords.words()]
    filtered_sentence = (" ").join(text_token_without_sw)
    c.append([filtered_sentence])
comment_words = ''
stopwords = set(STOPWORDS)
for i in pd.DataFrame(c)[0]:
    i = str(i)
    tokens = i.split()
    for i in range(len(tokens)):
        tokens[i]  = tokens[i].lower()
    comment_words += " ".join(tokens) + " "
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (12, 10), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
    
