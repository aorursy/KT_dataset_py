import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import pandas_profiling



import plotly

# plotly standard imports

import plotly.graph_objs as go

import plotly.plotly as py



# Cufflinks wrapper on plotly

import cufflinks as cf



# Options for pandas

#pd.options.display.max_columns = 30



# Display all cell outputs

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'



from plotly.offline import iplot, init_notebook_mode, plot

cf.go_offline()



init_notebook_mode(connected=True)



# Set global theme

cf.set_config_file(world_readable=True, theme='pearl')



import warnings  

warnings.filterwarnings('ignore')

np.seterr(divide='ignore', invalid='ignore')
df = pd.read_csv('../input/us-consumer-finance-complaints/consumer_complaints.csv',low_memory=False)

df_copy = df.copy() # Save a copy for later
df.head(10)
df.sample(5,random_state=89)
df.tail()
df.columns = df.columns.str.upper()

df.head(2)
df.isnull().mean().round(4)*100
# code chunk that I saw in Gabriel Preda kernel

# Reference write kernel here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def missing_values(data):

    total = data.isnull().sum().sort_values(ascending = False) # getting the sum of null values and ordering

    percent = (data.isnull().sum() / data.isnull().count() * 100 ).sort_values(ascending = False) #getting the percent and order of null

    df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) # Concatenating the total and percent

    print("Total columns at least one Values: ")

    print (df[~(df['Total'] == 0)]) # Returning values of nulls different of 0

    

    #print("\n Total of Sales % of Total: ", round((df[df['totals.transactionRevenue'] != np.nan]['totals.transactionRevenue'].count() / len(df_train['totals.transactionRevenue']) * 100),4))

    

    return 

missing_values(df)
# Function that takes the data type of a single column and converts it into easier to understand language

def get_var_category(series):

    unique_count = series.nunique(dropna=False)

    total_count = len(series)

    if pd.api.types.is_numeric_dtype(series):

        return 'Numerical'

    elif pd.api.types.is_datetime64_dtype(series):

        return 'Date'

    elif unique_count==total_count:

        return 'Text (Unique)'

    else:

        return 'Categorical'



def print_categories(df):

    for column_name in df.columns:

        print(column_name, ": ", get_var_category(df[column_name]))

print_categories(df)
df.memory_usage(deep=True).sum()

#df.info(memory_usage=True)
df['ISSUE'] = df.ISSUE.astype('category')

df['DATE_RECEIVED'] = df.DATE_RECEIVED.astype('category')

df['PRODUCT'] = df.PRODUCT.astype('category')

df['SUB_ISSUE'] = df.SUB_ISSUE.astype('category')

df['CONSUMER_COMPLAINT_NARRATIVE'] = df.CONSUMER_COMPLAINT_NARRATIVE.astype('category')

df['COMPANY_PUBLIC_RESPONSE'] = df.COMPANY_PUBLIC_RESPONSE.astype('category')

df['COMPANY'] = df.COMPANY.astype('category')

df['CONSUMER_CONSENT_PROVIDED'] = df.CONSUMER_CONSENT_PROVIDED.astype('category')

df['COMPANY_RESPONSE_TO_CONSUMER'] = df.COMPANY_RESPONSE_TO_CONSUMER.astype('category')

df['SUBMITTED_VIA'] = df.SUBMITTED_VIA.astype('category')
df.memory_usage(deep=True).sum()

#df.info(memory_usage=True)
#Witness the magic

pandas_profiling.ProfileReport(df)
#df.set_index(['ISSUE','PRODUCT','SUB_ISSUE','COMPANY','SUBMITTED_VIA'],inplace=True)

df[['ISSUE','DATE_RECEIVED','PRODUCT','SUB_ISSUE','CONSUMER_COMPLAINT_NARRATIVE',

    'COMPANY','COMPANY_PUBLIC_RESPONSE','CONSUMER_CONSENT_PROVIDED','COMPANY_RESPONSE_TO_CONSUMER','SUBMITTED_VIA']].describe().transpose()
import seaborn as sns; sns.set(style='white')

df['ISSUE'].str.strip("'").value_counts()[0:10].iplot(kind='bar',title='Top 10 issues',fontsize=14,color='#7070FF')
df['SUB_ISSUE'].str.strip("'").value_counts()[0:10].iplot(kind='bar',

                                                          title='Top 10 Sub Issues',fontsize=14,color='#9370DB')
from wordcloud import WordCloud, STOPWORDS



%matplotlib inline

text = df['PRODUCT'].values

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = '#F0F0F0',

    stopwords = STOPWORDS).generate(str(text))



fig = plt.figure(

    figsize = (14, 10),

    facecolor = '#F0F0F0',

    edgecolor = '#F0F0F0')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)
comp_dist = df['COMPANY'].str.strip("'").value_counts()[0:10]

fig = {

  "data": [

    {

      "values": comp_dist.values,

      "labels": comp_dist.index

      ,

      "domain": {"column": 0},

      "name": "Bank Complaints",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    },

    ],

  "layout": {

        "title":"Top 10 Banks receiving most complaints",

        "grid": {"rows": 1, "columns": 1},

        "annotations": [

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "BANKS",

                "x": 0.5,

                "y": 0.5

            }

        ]

    }

}

iplot(fig)
pd.crosstab(df['TIMELY_RESPONSE'],df['SUBMITTED_VIA']).sort_index().iplot(kind='bar',barmode='stack')
states = df['STATE'].value_counts()



scl = [

    [0.0, 'rgb(242,240,247)'],

    [0.2, 'rgb(218,218,235)'],

    [0.4, 'rgb(188,189,220)'],

    [0.6, 'rgb(158,154,200)'],

    [0.8, 'rgb(117,107,177)'],

    [1.0, 'rgb(84,39,143)']

]



data = [go.Choropleth(

    colorscale = scl,

    autocolorscale = False,

    locations = states.index,

    z = states.values,

    locationmode = 'USA-states',

    text = states.index,

    marker = go.choropleth.Marker(

        line = go.choropleth.marker.Line(

            color = 'rgb(255,255,255)',

            width = 2

        )),

    colorbar = go.choropleth.ColorBar(

        title = "Complaints")

)]



layout = go.Layout(

    title = go.layout.Title(

        text = 'Complaints by State<br>(Hover for breakdown)'

    ),

    geo = go.layout.Geo(

        scope = 'usa',

        projection = go.layout.geo.Projection(type = 'albers usa'),

        showlakes = True,

        lakecolor = 'rgb(100,149,237)'),

)



fig = go.Figure(data = data, layout = layout)

iplot(fig)
# library of datetime

from datetime import datetime
df["DATE"] = pd.to_datetime(df["DATE_RECEIVED"]) # seting the column as pandas datetime

df["YEAR"] = df['DATE'].dt.year # extracting year

df["MONTH"] = df["DATE"].dt.month # extracting month

df["WEEKDAY_NAME"] = df["DATE"].dt.weekday_name # extracting name of the weekday

df["YEAR_MONTH"] = df.YEAR.astype(str).str.cat(df.MONTH.astype(str), sep='-')
df["YEAR_MONTH"].value_counts().iplot(kind='bar',color='#800000',title='Number of Complaints per Month')
df["WEEKDAY_NAME"].value_counts().iplot(kind='barh',title='Number of Complaints per Weekday',color='cornflowerblue')
prod_dist = df.groupby(['COMPANY_RESPONSE_TO_CONSUMER']).size()

trace = go.Pie(labels=prod_dist.index, values=prod_dist,title='Company Response to the Customer')

iplot([trace])
test = pd.crosstab(df['TIMELY_RESPONSE'],df['CONSUMER_DISPUTED?'])#.apply(lambda x: x/x.sum() * 100).astype(int)

cm = sns.light_palette("blue", as_cmap=True)



test.style.background_gradient(cmap=cm)
pd.crosstab(df['TIMELY_RESPONSE'],df['CONSUMER_DISPUTED?']).iplot(kind='bar',title='Timely Response vs Consumer Disputed')